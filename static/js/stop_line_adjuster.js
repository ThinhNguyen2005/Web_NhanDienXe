/**
 * Stop Line Adjuster - Module cho phép hiệu chỉnh vạch dừng và vùng ROI bằng tay
 */

class RoiAdjuster {
  constructor(canvasId, videoId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.video = document.getElementById(videoId);

    // Vùng ROI
    this.waitingZone = []; // Mảng điểm [(x,y), ...]
    this.violationZone = []; // Mảng điểm [(x,y), ...]

    // Trạng thái vẽ
    this.isDrawingWaiting = false;
    this.isDrawingViolation = false;

    this.setupEventListeners();
    this.updateCanvas();
  }

  setupEventListeners() {
    // Hàm tiện ích để gán sự kiện một cách an toàn
    const addListener = (id, event, handler) => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener(event, handler);
        } else {
            // Nếu không tìm thấy, báo lỗi trong console thay vì làm sập ứng dụng
            console.error(`Lỗi Javascript: Không tìm thấy phần tử với ID '${id}' trong file HTML.`);
        }
    };

    // Xử lý click chuột trên canvas
    if (this.canvas) {
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    }

    // Gán sự kiện cho các nút bằng hàm an toàn
    addListener('btnDrawWaitingZone', 'click', () => {
        this.isDrawingWaiting = true;
        this.isDrawingViolation = false;
        this.waitingZone = [];
        this.updateCanvas();
    });
    
    addListener('btnDrawViolationZone', 'click', () => {
        this.isDrawingWaiting = false;
        this.isDrawingViolation = true;
        this.violationZone = [];
        this.updateCanvas();
    });
    
    addListener('btnSaveROI', 'click', () => {
        this.saveROI();
    });
    // Khởi tạo kích thước canvas khi video được tải
    if (this.video) {
        this.video.addEventListener('loadedmetadata', () => {
            if (this.video.videoWidth > 0) {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
            }
            this.updateCanvas();
        });
    }
  }

  handleMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Xử lý vẽ vùng ROI
    if (this.isDrawingWaiting) {
      this.waitingZone.push([x, y]);
    } else if (this.isDrawingViolation) {
      this.violationZone.push([x, y]);
    }

    this.updateCanvas();
  }

  updateCanvas() {
    if (!this.canvas || !this.ctx) return;

    // Xóa canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Vẽ frame từ video nếu có
    if (this.video && this.video.readyState >= 2) {
      this.ctx.drawImage(
        this.video,
        0,
        0,
        this.canvas.width,
        this.canvas.height
      );
    }

    // Vẽ vùng chờ (màu vàng nhạt)
    this.ctx.fillStyle = "rgba(255, 255, 0, 0.3)";
    if (this.waitingZone.length > 2) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.waitingZone[0][0], this.waitingZone[0][1]);
      for (let i = 1; i < this.waitingZone.length; i++) {
        this.ctx.lineTo(this.waitingZone[i][0], this.waitingZone[i][1]);
      }
      this.ctx.closePath();
      this.ctx.fill();
      this.ctx.strokeStyle = "yellow";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    }

    // Vẽ vùng vi phạm (màu đỏ nhạt)
    this.ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
    if (this.violationZone.length > 2) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.violationZone[0][0], this.violationZone[0][1]);
      for (let i = 1; i < this.violationZone.length; i++) {
        this.ctx.lineTo(this.violationZone[i][0], this.violationZone[i][1]);
      }
      this.ctx.closePath();
      this.ctx.fill();
      this.ctx.strokeStyle = "red";
      this.ctx.lineWidth = 2;
      this.ctx.stroke();
    }

    // Vẽ các điểm
    const drawPoints = (points, color) => {
      points.forEach((point, i) => {
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(point[0], point[1], 5, 0, Math.PI * 2);
        this.ctx.fill();

        // Hiển thị số thứ tự điểm
        this.ctx.fillStyle = "white";
        this.ctx.font = "12px Arial";
        this.ctx.fillText(i + 1, point[0] + 8, point[1] - 8);
      });
    };

    drawPoints(this.waitingZone, "yellow");
    drawPoints(this.violationZone, "red");
  }

  saveROI() {
    // Kiểm tra số điểm tối thiểu
    if (this.waitingZone.length < 3 || this.violationZone.length < 3) {
      alert("Vùng chờ và vùng vi phạm phải có ít nhất 3 điểm!");
      return;
    }

    // Chuẩn bị dữ liệu
    const data = {
      camera_id: document.getElementById("cameraId").value || "default",
      waiting_zone: this.waitingZone,
      violation_zone: this.violationZone,
    };

    // Gửi API request để lưu ROI
    fetch("/api/save_roi", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      // Mã mới trong stop_line_adjuster.js
      // ...
      .then((data) => {
        if (data.success) {
          alert("Đã lưu ROI thành công!");
          // --- THÊM DÒNG NÀY ---
          // Kích hoạt nút xử lý video sau khi lưu ROI thành công
          const btnProcessVideo = document.getElementById("btnProcessVideo");
          if (btnProcessVideo) {
            btnProcessVideo.disabled = false;
            btnProcessVideo.removeAttribute("title");
          }
        } else {
          // ...
          alert("Lỗi khi lưu ROI: " + data.error);
        }
      })
      .catch((error) => {
        console.error("Lỗi:", error);
        alert("Lỗi khi lưu ROI");
      });
  }

  reset() {
    this.waitingZone = [];
    this.violationZone = [];
    this.isDrawingWaiting = false;
    this.isDrawingViolation = false;
    this.updateCanvas();
  }

  loadROI(cameraId) {
    fetch(`/api/load_roi/${cameraId}`)
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          this.waitingZone = data.data.waiting_zone;
          this.violationZone = data.data.violation_zone;
          this.updateCanvas();
        } else {
          console.error("Lỗi khi tải ROI:", data.error);
        }
      })
      .catch((error) => {
        console.error("Lỗi khi tải ROI:", error);
      });
  }
}