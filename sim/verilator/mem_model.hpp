// QMEM bus model: the external memory the core streams from (docs/RTL.md 2.1).
// Fixed read latency, one returned beat per --bw-div cycles, in-order returns
// across tags, a 64-beat in-flight window that throttles rd_req_ready, byte
// strobed writes and their acks after the same latency. Contents are the
// compiled image.bin mapped copy-on-write, so a write lands in this run only.
// A read takes its bytes when the request is accepted and holds them in the
// pending beat, so a write accepted while the burst is in flight leaves the
// returned data alone; reading memory the core has not fenced against reads
// what was there at the request.
// Everything is driven for the next rising edge and sampled after it, the same
// discipline as the cocotb model in sim/cocotb/qc_qmem.py.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qcore {

constexpr uint64_t MEM_PAGE = 4096;

// Byte-addressed memory: image.bin mapped MAP_PRIVATE, with pages beyond the
// image held in an overlay so a stray address is stored rather than dropped.
class MemBytes {
 public:
  ~MemBytes() {
    if (base_ != nullptr) munmap(base_, static_cast<size_t>(size_));
    if (fd_ >= 0) close(fd_);
  }

  void open_image(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    if (fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat " + path);
    size_ = static_cast<uint64_t>(st.st_size);
    void* p = mmap(nullptr, static_cast<size_t>(size_), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
    base_ = static_cast<uint8_t*>(p);
  }

  uint64_t size() const { return size_; }

  void read(uint64_t addr, uint32_t n, uint8_t* dst) {
    if (addr + n <= size_) {
      memcpy(dst, base_ + addr, n);
      return;
    }
    for (uint32_t i = 0; i < n; i++) dst[i] = read_byte(addr + i);
  }

  void write(uint64_t addr, uint32_t n, const uint8_t* src) {
    if (addr + n <= size_) {
      memcpy(base_ + addr, src, n);
      return;
    }
    for (uint32_t i = 0; i < n; i++) write_byte(addr + i, src[i]);
  }

  uint8_t read_byte(uint64_t addr) {
    if (addr < size_) return base_[addr];
    auto it = overlay_.find(addr / MEM_PAGE);
    if (it == overlay_.end()) return 0;
    return it->second[addr % MEM_PAGE];
  }

  void write_byte(uint64_t addr, uint8_t v) {
    if (addr < size_) {
      base_[addr] = v;
      return;
    }
    auto& page = overlay_[addr / MEM_PAGE];
    if (page.empty()) page.assign(MEM_PAGE, 0);
    page[addr % MEM_PAGE] = v;
  }

 private:
  int fd_ = -1;
  uint8_t* base_ = nullptr;
  uint64_t size_ = 0;
  std::unordered_map<uint64_t, std::vector<uint8_t>> overlay_;
};

// The bus itself, on a DUT with the QMEM ports of docs/RTL.md 2.1.
template <class Dut, int WB>
class Qmem {
 public:
  Qmem(Dut* dut, MemBytes* mem, uint32_t latency, uint32_t bw_div)
      : dut_(dut), mem_(mem), latency_(latency), bw_div_(bw_div ? bw_div : 1) {}

  uint64_t rd_beats = 0, rd_bytes = 0, wr_beats = 0, wr_bytes = 0, rd_requests = 0;

  void reset_ports() {
    dut_->rd_req_ready = 0;
    dut_->rd_data_valid = 0;
    dut_->rd_data_tag = 0;
    dut_->rd_data_last = 0;
    dut_->wr_ready = 0;
    dut_->wr_ack = 0;
    memset(&dut_->rd_data, 0, sizeof(dut_->rd_data));
    prev_ = Sample();
  }

  uint64_t outstanding() const { return pending_.size(); }
  uint64_t writes_outstanding() const { return acks_.size(); }

  // Commit the transfers of the edge just passed, then drive the ports that
  // the DUT samples at the next one.
  void drive(uint64_t now) {
    t_ = now;
    if (prev_.rd_valid && prev_.rd_ready) accept_read();
    if (prev_.wr_valid && prev_.wr_ready) accept_write();
    const uint64_t nxt = t_ + 1;
    if (!pending_.empty() && pending_.front().deliver <= nxt) {
      const Beat& b = pending_.front();
      memcpy(&dut_->rd_data, b.data, WB);
      dut_->rd_data_valid = 1;
      dut_->rd_data_tag = b.tag;
      dut_->rd_data_last = b.last ? 1 : 0;
      pending_.pop_front();
      rd_beats++;
      rd_bytes += WB;
    } else {
      dut_->rd_data_valid = 0;
      dut_->rd_data_last = 0;
    }
    if (!acks_.empty() && acks_.front() <= nxt) {
      acks_.pop_front();
      dut_->wr_ack = 1;
    } else {
      dut_->wr_ack = 0;
    }
    rd_ready_ = pending_.size() < WINDOW;
    dut_->rd_req_ready = rd_ready_ ? 1 : 0;
    dut_->wr_ready = 1;
  }

  // Record what the DUT presents for the next rising edge.
  void sample() {
    Sample s;
    s.rd_valid = dut_->rd_req_valid != 0;
    if (s.rd_valid) {
      s.rd_addr = static_cast<uint64_t>(dut_->rd_req_addr);
      s.rd_len = static_cast<uint32_t>(dut_->rd_req_len);
      s.rd_tag = static_cast<uint32_t>(dut_->rd_req_tag);
    }
    s.wr_valid = dut_->wr_valid != 0;
    if (s.wr_valid) {
      s.wr_addr = static_cast<uint64_t>(dut_->wr_addr);
      memcpy(s.wr_data, &dut_->wr_data, WB);
      memcpy(s.wr_strb, &dut_->wr_strb, (WB + 7) / 8);
    }
    s.rd_ready = rd_ready_;
    s.wr_ready = true;
    prev_ = s;
  }

 private:
  static constexpr size_t WINDOW = 64;

  // The bytes are the memory as it stood when the request was accepted.
  struct Beat {
    uint64_t deliver;
    uint32_t tag;
    bool last;
    uint8_t data[WB];
  };
  struct Sample {
    bool rd_valid = false, rd_ready = false, wr_valid = false, wr_ready = false;
    uint64_t rd_addr = 0, wr_addr = 0;
    uint32_t rd_len = 0, rd_tag = 0;
    uint8_t wr_data[WB] = {};
    uint8_t wr_strb[(WB + 7) / 8] = {};
  };

  void accept_read() {
    rd_requests++;
    for (uint32_t i = 0; i < prev_.rd_len; i++) {
      uint64_t deliver = t_ + latency_ + i;
      if (last_deliver_ >= 0 && deliver < static_cast<uint64_t>(last_deliver_) + bw_div_) {
        deliver = static_cast<uint64_t>(last_deliver_) + bw_div_;
      }
      last_deliver_ = static_cast<int64_t>(deliver);
      pending_.push_back(Beat{deliver, prev_.rd_tag, i + 1 == prev_.rd_len, {}});
      mem_->read(prev_.rd_addr + static_cast<uint64_t>(i) * WB, WB, pending_.back().data);
    }
  }

  void accept_write() {
    uint32_t strobed = 0;
    for (int j = 0; j < WB; j++) {
      if ((prev_.wr_strb[j / 8] >> (j % 8)) & 1) {
        mem_->write_byte(prev_.wr_addr + static_cast<uint64_t>(j), prev_.wr_data[j]);
        strobed++;
      }
    }
    wr_beats++;
    wr_bytes += strobed;
    acks_.push_back(t_ + latency_);
  }

  Dut* dut_;
  MemBytes* mem_;
  uint32_t latency_;
  uint32_t bw_div_;
  uint64_t t_ = 0;
  int64_t last_deliver_ = -1;
  bool rd_ready_ = false;
  std::deque<Beat> pending_;
  std::deque<uint64_t> acks_;
  Sample prev_;
};

}  // namespace qcore
