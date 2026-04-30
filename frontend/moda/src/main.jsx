import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { motion, useScroll, useTransform } from "framer-motion";
import {
  AlertTriangle,
  ArrowDownToLine,
  CheckCircle2,
  Clock3,
  Database,
  Eye,
  FileVideo,
  History,
  Loader2,
  Play,
  Radar,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  Upload,
  Waypoints
} from "lucide-react";
import "./styles.css";

const heroImage =
  "https://images.unsplash.com/photo-1494526585095-c41746248156?auto=format&fit=crop&w=2200&q=85";

const navItems = [
  ["Upload", "upload"],
  ["ROI", "roi"],
  ["Processing", "processing"],
  ["Results", "results"],
  ["Search", "search"],
  ["History", "history"]
];

function api(path, options) {
  return fetch(path, options).then(async (response) => {
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.success === false) {
      throw new Error(data.error || data.message || "Request failed");
    }
    return data;
  });
}

function Reveal({ children, className = "", delay = 0 }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.18 }}
      transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1], delay }}
    >
      {children}
    </motion.div>
  );
}

function Header({ activeJobId }) {
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-white/30 bg-ivory/80 backdrop-blur-xl">
      <nav className="mx-auto flex h-20 max-w-7xl items-center justify-between px-5 sm:px-8">
        <a href="#home" className="font-display text-2xl tracking-[0.14em] text-ink sm:text-3xl">
          REDLIGHT AI
        </a>
        <div className="hidden items-center gap-7 text-xs uppercase tracking-[0.25em] text-cocoa/80 lg:flex">
          {navItems.map(([label, id]) => (
            <a key={id} href={`#${id}`}>
              {label}
            </a>
          ))}
        </div>
        <a href={activeJobId ? "#processing" : "#upload"} className="quiet-button">
          {activeJobId ? "Theo dõi" : "Bắt đầu"}
        </a>
      </nav>
    </header>
  );
}

function Hero({ stats }) {
  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 0.45], [0, 100]);

  return (
    <section id="home" className="relative min-h-screen overflow-hidden">
      <motion.div style={{ y }} className="absolute inset-0 scale-110 bg-cover bg-center blur-[2px]">
        <div className="h-full w-full bg-cover bg-center" style={{ backgroundImage: `url(${heroImage})` }} />
      </motion.div>
      <div className="absolute inset-0 bg-gradient-to-b from-ivory/88 via-linen/76 to-ivory" />
      <div className="relative z-10 mx-auto flex min-h-screen max-w-7xl items-end px-5 pb-16 pt-32 sm:px-8 lg:pb-24">
        <Reveal className="max-w-6xl">
          <p className="mb-7 text-xs uppercase tracking-[0.48em] text-cocoa/80">Traffic violation intelligence studio</p>
          <h1 className="font-display text-[4.2rem] leading-[0.9] text-ink sm:text-[7rem] lg:text-[10rem]">
            Nhận diện vượt đèn đỏ, rõ ràng và tinh gọn.
          </h1>
          <div className="mt-10 grid max-w-4xl gap-5 text-cocoa/85 md:grid-cols-[1.3fr_1fr]">
            <p className="text-base leading-8 sm:text-lg">
              Upload video, vẽ vùng ROI, xử lý bằng YOLO và đọc biển số xe vi phạm trong một giao diện tối giản, nhẹ hơn và dễ theo dõi hơn.
            </p>
            <div className="grid grid-cols-3 gap-3">
              <Metric label="Videos" value={stats.videos} />
              <Metric label="Jobs" value={stats.jobs} />
              <Metric label="Violations" value={stats.violations} />
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-md border border-white/60 bg-ivory/55 p-4 text-center shadow-soft backdrop-blur">
      <div className="font-display text-4xl text-ink">{value}</div>
      <div className="mt-1 text-[0.65rem] uppercase tracking-[0.25em] text-cocoa/70">{label}</div>
    </div>
  );
}

function WaveDivider() {
  return (
    <div className="-mt-1 overflow-hidden bg-ivory">
      <motion.svg
        viewBox="0 0 1440 160"
        className="h-24 w-[160%] -translate-x-[10%] text-linen sm:h-32"
        preserveAspectRatio="none"
        initial={{ x: -80 }}
        whileInView={{ x: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 1.2, ease: "easeOut" }}
      >
        <motion.path
          fill="currentColor"
          d="M0,64 C210,128 320,8 520,54 C725,102 820,160 1040,98 C1220,48 1320,72 1440,24 L1440,160 L0,160 Z"
          animate={{
            d: [
              "M0,64 C210,128 320,8 520,54 C725,102 820,160 1040,98 C1220,48 1320,72 1440,24 L1440,160 L0,160 Z",
              "M0,82 C220,28 360,124 548,72 C724,22 850,84 1048,66 C1235,48 1325,126 1440,52 L1440,160 L0,160 Z",
              "M0,64 C210,128 320,8 520,54 C725,102 820,160 1040,98 C1220,48 1320,72 1440,24 L1440,160 L0,160 Z"
            ]
          }}
          transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.svg>
    </div>
  );
}

function UploadSection({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [state, setState] = useState({ loading: false, message: "" });

  async function submit(event) {
    event.preventDefault();
    if (!file) return setState({ loading: false, message: "Vui lòng chọn video trước." });
    const formData = new FormData();
    formData.append("video", file);
    setState({ loading: true, message: "Đang tải video lên..." });
    try {
      const data = await api("/api/upload", { method: "POST", body: formData });
      onUploaded(data);
      setState({ loading: false, message: `Đã tải lên ${data.filename}. Tiếp theo hãy vẽ ROI.` });
      window.location.hash = "roi";
    } catch (error) {
      setState({ loading: false, message: error.message });
    }
  }

  return (
    <section id="upload" className="bg-linen px-5 py-24 sm:px-8 lg:py-32">
      <div className="mx-auto grid max-w-7xl gap-10 lg:grid-cols-[0.85fr_1.15fr]">
        <Reveal>
          <p className="section-kicker">01 Upload</p>
          <h2 className="section-title">Đưa video giao lộ vào hệ thống.</h2>
          <p className="mt-7 max-w-xl text-lg leading-9 text-cocoa/80">
            Hỗ trợ MP4, AVI, MOV, MKV. Sau khi upload, bạn sẽ thiết lập vùng chờ và vùng vi phạm ngay trong giao diện mới.
          </p>
        </Reveal>
        <Reveal delay={0.08}>
          <form onSubmit={submit} className="panel p-6 sm:p-8">
            <label className="block text-xs uppercase tracking-[0.3em] text-cocoa/65">Video file</label>
            <input
              className="mt-5 w-full rounded-md border border-oat/60 bg-ivory/80 p-4 text-cocoa file:mr-5 file:rounded-full file:border-0 file:bg-cocoa file:px-5 file:py-2 file:text-ivory"
              type="file"
              accept="video/mp4,video/avi,video/mov,video/mkv"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <button className="primary-button mt-6 w-full" disabled={state.loading}>
              {state.loading ? <Loader2 className="animate-spin" size={18} /> : <Upload size={18} />} Tải lên video
            </button>
            {state.message && <p className="mt-5 text-sm leading-7 text-cocoa/75">{state.message}</p>}
          </form>
        </Reveal>
      </div>
    </section>
  );
}

function RoiSection({ videos, selectedVideo, setSelectedVideo, cameraId, setCameraId, processingOptions, setProcessingOptions, onProcess }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [mode, setMode] = useState("waiting");
  const [waitingZone, setWaitingZone] = useState([]);
  const [violationZone, setViolationZone] = useState([]);
  const [message, setMessage] = useState("");

  const selected = useMemo(() => videos.find((item) => item.path === selectedVideo), [videos, selectedVideo]);
  const videoUrl = selectedVideo ? `/api/get_video/${selectedVideo}` : "";

  useEffect(() => {
    const cam = selected?.camera_id || "";
    setCameraId(cam);
    if (cam) {
      api(`/api/load_roi/${encodeURIComponent(cam)}`)
        .then((data) => {
          setWaitingZone(data.data?.waiting_zone || []);
          setViolationZone(data.data?.violation_zone || []);
        })
        .catch(() => {
          setWaitingZone([]);
          setViolationZone([]);
        });
    }
  }, [selectedVideo]);

  useEffect(() => {
    drawCanvas();
  }, [waitingZone, violationZone, mode, selectedVideo]);

  function prepareCanvas() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    drawCanvas();
  }

  function drawCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawPolygon(ctx, waitingZone, "rgba(185, 143, 67, 0.35)", "#b98f43", mode === "waiting");
    drawPolygon(ctx, violationZone, "rgba(150, 63, 45, 0.33)", "#963f2d", mode === "violation");
  }

  function drawPolygon(ctx, points, fill, stroke, active) {
    if (!points.length) return;
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = active ? 4 : 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach((point) => ctx.lineTo(point[0], point[1]));
    if (points.length >= 3) ctx.closePath();
    ctx.fill();
    ctx.stroke();
    points.forEach((point, index) => {
      ctx.fillStyle = stroke;
      ctx.beginPath();
      ctx.arc(point[0], point[1], 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fbf7ef";
      ctx.font = "14px Inter";
      ctx.fillText(String(index + 1), point[0] + 10, point[1] - 8);
    });
  }

  function addPoint(event) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.round((event.clientX - rect.left) * (canvas.width / rect.width));
    const y = Math.round((event.clientY - rect.top) * (canvas.height / rect.height));
    if (mode === "waiting") setWaitingZone((points) => [...points, [x, y]]);
    else setViolationZone((points) => [...points, [x, y]]);
  }

  async function saveRoi() {
    if (!cameraId) return setMessage("Vui lòng chọn video hoặc nhập camera ID.");
    if (waitingZone.length < 3 || violationZone.length < 3) {
      return setMessage("Mỗi vùng ROI cần ít nhất 3 điểm.");
    }
    try {
      await api("/api/save_roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: cameraId, waiting_zone: waitingZone, violation_zone: violationZone })
      });
      setMessage("Đã lưu ROI. Bạn có thể bắt đầu xử lý video.");
    } catch (error) {
      setMessage(error.message);
    }
  }

  function processSelected() {
    if (!selected || selected.source !== "uploads") return setMessage("Vui lòng chọn video đã upload.");
    onProcess(selected.name, processingOptions);
  }

  return (
    <section id="roi" className="bg-ivory px-5 py-24 sm:px-8 lg:py-32">
      <div className="mx-auto max-w-7xl">
        <Reveal className="mb-12">
          <p className="section-kicker">02 ROI</p>
          <h2 className="section-title max-w-5xl">Vẽ vùng chờ và vùng vi phạm bằng thao tác trực tiếp.</h2>
        </Reveal>
        <div className="grid gap-8 lg:grid-cols-[1.35fr_0.65fr]">
          <Reveal>
            <div className="panel overflow-hidden p-4">
              <div className="relative aspect-video overflow-hidden rounded-md bg-ink/10">
                {selectedVideo ? (
                  <>
                    <video ref={videoRef} src={videoUrl} controls onLoadedMetadata={prepareCanvas} className="h-full w-full object-contain" />
                    <canvas ref={canvasRef} onClick={addPoint} className="absolute inset-0 h-full w-full cursor-crosshair" />
                  </>
                ) : (
                  <div className="flex h-full items-center justify-center text-cocoa/65">Chọn hoặc upload video để thiết lập ROI</div>
                )}
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.08}>
            <div className="panel space-y-5 p-6">
              <label className="field-label">Video</label>
              <select className="field" value={selectedVideo} onChange={(event) => setSelectedVideo(event.target.value)}>
                <option value="">Chọn video</option>
                {videos.map((video) => (
                  <option key={`${video.source}-${video.path}`} value={video.path}>
                    {video.name}
                  </option>
                ))}
              </select>
              <label className="field-label">Camera ID</label>
              <input className="field" value={cameraId} onChange={(event) => setCameraId(event.target.value)} />
              <div className="grid grid-cols-2 gap-3">
                <button className={`soft-button ${mode === "waiting" ? "is-active" : ""}`} onClick={() => setMode("waiting")}>
                  Vùng chờ
                </button>
                <button className={`soft-button ${mode === "violation" ? "is-active" : ""}`} onClick={() => setMode("violation")}>
                  Vùng vi phạm
                </button>
              </div>
              <button className="soft-button w-full" onClick={() => (mode === "waiting" ? setWaitingZone([]) : setViolationZone([]))}>
                Xóa vùng đang vẽ
              </button>
              <label className="field-label">Chế độ xử lý</label>
              <select
                className="field"
                value={processingOptions.mode}
                onChange={(event) => {
                  const modeValue = event.target.value;
                  setProcessingOptions((current) => ({
                    ...current,
                    mode: modeValue,
                    write_output_video: modeValue === "fast" ? false : current.write_output_video
                  }));
                }}
              >
                <option value="fast">Fast - nhanh, ít ghi output</option>
                <option value="balanced">Balanced - cân bằng</option>
                <option value="quality">Quality - kỹ hơn</option>
              </select>
              <label className="flex items-center gap-3 rounded-md border border-oat/40 bg-ivory/55 p-4 text-sm text-cocoa">
                <input
                  type="checkbox"
                  checked={processingOptions.write_output_video}
                  onChange={(event) => setProcessingOptions((current) => ({ ...current, write_output_video: event.target.checked }))}
                />
                Ghi video đã xử lý để tải về
              </label>
              <button className="primary-button w-full" onClick={saveRoi}>
                <CheckCircle2 size={18} /> Lưu ROI
              </button>
              <button className="primary-button w-full" onClick={processSelected}>
                <Play size={18} /> Xử lý video
              </button>
              <p className="text-sm leading-7 text-cocoa/70">
                Vùng chờ: {waitingZone.length} điểm. Vùng vi phạm: {violationZone.length} điểm.
              </p>
              {message && <p className="rounded-md bg-linen/70 p-4 text-sm leading-7 text-cocoa">{message}</p>}
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

function ProcessingSection({ activeJobId, status, activeFilename }) {
  const progress = Math.round(status?.progress || 0);
  return (
    <section id="processing" className="bg-[#e6d4bf] px-5 py-24 sm:px-8 lg:py-32">
      <div className="mx-auto max-w-7xl">
        <Reveal className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
          <div>
            <p className="section-kicker">03 Processing</p>
            <h2 className="section-title">Theo dõi xử lý nền.</h2>
          </div>
          <div className="panel p-7 sm:p-9">
            {activeJobId ? (
              <>
                <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                  <div>
                    <p className="text-xs uppercase tracking-[0.25em] text-cocoa/60">Job</p>
                    <h3 className="mt-2 font-display text-4xl text-ink">{activeJobId}</h3>
                    <p className="mt-2 text-cocoa/70">{activeFilename}</p>
                  </div>
                  <div className="rounded-full border border-cocoa/20 px-5 py-3 text-sm uppercase tracking-[0.2em] text-cocoa">
                    {status?.status || "starting"}
                  </div>
                </div>
                <div className="mt-8 h-3 overflow-hidden rounded-full bg-ivory/70">
                  <motion.div className="h-full bg-cocoa" animate={{ width: `${progress}%` }} />
                </div>
                <div className="mt-6 grid gap-4 sm:grid-cols-3">
                  <StatusCard icon={<Clock3 />} label="Tiến độ" value={`${progress}%`} />
                  <StatusCard icon={<AlertTriangle />} label="Vi phạm" value={status?.violations_found || 0} />
                  <StatusCard icon={<Radar />} label="Trạng thái" value={status?.status || "starting"} />
                </div>
                {status?.options && (
                  <div className="mt-5 rounded-md border border-ivory/60 bg-ivory/35 p-4 text-sm leading-7 text-cocoa/75">
                    Mode: {status.options.mode || "balanced"} · Ghi video: {status.options.write_output_video ? "có" : "không"} ·
                    Vehicle interval: {status.options.vehicle_detection_interval} · Light interval: {status.options.traffic_light_interval}
                  </div>
                )}
                {status?.timing && (
                  <div className="mt-3 rounded-md border border-ivory/60 bg-ivory/25 p-4 text-sm leading-7 text-cocoa/70">
                    Timing: vehicle {status.timing.vehicle || 0}s · light {status.timing.light || 0}s · OCR {status.timing.ocr || 0}s · draw/encode {status.timing.draw_encode || 0}s
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center gap-4 text-cocoa/75">
                <Loader2 size={22} /> Chưa có job đang xử lý. Hãy upload video, lưu ROI rồi bấm xử lý.
              </div>
            )}
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function StatusCard({ icon, label, value }) {
  return (
    <div className="rounded-md border border-ivory/60 bg-ivory/45 p-5">
      <div className="text-cocoa">{React.cloneElement(icon, { size: 20 })}</div>
      <div className="mt-4 text-xs uppercase tracking-[0.22em] text-cocoa/60">{label}</div>
      <div className="mt-2 font-display text-3xl text-ink">{value}</div>
    </div>
  );
}

function ResultsSection({ results, onRefresh }) {
  return (
    <section id="results" className="bg-ivory px-5 py-24 sm:px-8 lg:py-32">
      <div className="mx-auto max-w-7xl">
        <Reveal className="mb-12 flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <div>
            <p className="section-kicker">04 Results</p>
            <h2 className="section-title">Kết quả vi phạm.</h2>
          </div>
          <button className="quiet-button w-fit" onClick={onRefresh}>
            <RefreshCw size={16} /> Cập nhật
          </button>
        </Reveal>
        <div className="grid gap-5">
          {results?.download_url && (
            <a className="primary-button w-fit" href={results.download_url}>
              <ArrowDownToLine size={18} /> Tải video đã xử lý
            </a>
          )}
          {results?.violations?.length ? (
            <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
              {results.violations.map((violation, index) => (
                <motion.article key={`${violation.id || index}-${violation.frame_number}`} className="panel overflow-hidden" whileHover={{ y: -5 }}>
                  <img src={violation.image_url} alt="Ảnh vi phạm" className="aspect-video w-full object-cover" />
                  <div className="p-6">
                    <div className="font-display text-4xl text-ink">{violation.license_plate || "UNKNOWN"}</div>
                    <p className="mt-3 text-sm leading-7 text-cocoa/75">
                      Frame {violation.frame_number || "N/A"} · Tin cậy {Math.round((violation.confidence || 0) * 100)}%
                    </p>
                  </div>
                </motion.article>
              ))}
            </div>
          ) : (
            <div className="panel p-8 text-cocoa/75">Chưa có kết quả hoặc job không phát hiện vi phạm.</div>
          )}
        </div>
      </div>
    </section>
  );
}

function SearchSection() {
  const [plate, setPlate] = useState("");
  const [rows, setRows] = useState([]);
  const [message, setMessage] = useState("");

  async function submit(event) {
    event.preventDefault();
    try {
      const data = await api(`/api/search?plate=${encodeURIComponent(plate)}`);
      setRows(data.violations || []);
      setMessage(`Tìm thấy ${(data.violations || []).length} kết quả.`);
    } catch (error) {
      setMessage(error.message);
    }
  }

  return (
    <section id="search" className="bg-linen px-5 py-24 sm:px-8 lg:py-32">
      <div className="mx-auto max-w-7xl">
        <Reveal className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
          <div>
            <p className="section-kicker">05 Search</p>
            <h2 className="section-title">Tra cứu biển số.</h2>
          </div>
          <div className="panel p-7">
            <form className="flex flex-col gap-3 sm:flex-row" onSubmit={submit}>
              <input className="field flex-1" value={plate} onChange={(event) => setPlate(event.target.value)} placeholder="Ví dụ: 30E, 59A-123.45" />
              <button className="primary-button">
                <Search size={18} /> Tìm
              </button>
            </form>
            {message && <p className="mt-5 text-sm text-cocoa/70">{message}</p>}
            <div className="mt-6 overflow-x-auto">
              <table className="minimal-table">
                <thead>
                  <tr>
                    <th>Biển số</th>
                    <th>Frame</th>
                    <th>Tin cậy</th>
                    <th>Thời gian</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.id}>
                      <td>{row.license_plate}</td>
                      <td>{row.frame_number}</td>
                      <td>{Math.round((row.confidence || 0) * 100)}%</td>
                      <td>{String(row.timestamp || "").replace("T", " ").slice(0, 19)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function HistorySection({ history, onLoadHistory, onOpenResults }) {
  async function remove(jobId) {
    if (!window.confirm(`Xóa lịch sử ${jobId}?`)) return;
    await api(`/api/history/${encodeURIComponent(jobId)}/delete`, { method: "POST" });
    onLoadHistory();
  }

  return (
    <section id="history" className="bg-ivory px-5 py-24 sm:px-8 lg:py-32">
      <div className="mx-auto max-w-7xl">
        <Reveal className="mb-12 flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <div>
            <p className="section-kicker">06 History</p>
            <h2 className="section-title">Lịch sử xử lý.</h2>
          </div>
          <button className="quiet-button w-fit" onClick={onLoadHistory}>
            <RefreshCw size={16} /> Tải lại
          </button>
        </Reveal>
        <div className="panel overflow-x-auto p-4">
          <table className="minimal-table">
            <thead>
              <tr>
                <th>Job</th>
                <th>Vi phạm</th>
                <th>Video</th>
                <th>Thao tác</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.job_id}>
                  <td>{item.job_id}</td>
                  <td>{item.violation_count}</td>
                  <td>{item.output_video || "N/A"}</td>
                  <td>
                    <div className="flex flex-wrap gap-2">
                      <button className="table-action" onClick={() => onOpenResults(item.job_id)}>
                        <Eye size={15} /> Xem
                      </button>
                      {item.processed_video_url && (
                        <a className="table-action" href={item.processed_video_url}>
                          <ArrowDownToLine size={15} /> Tải
                        </a>
                      )}
                      <button className="table-action danger" onClick={() => remove(item.job_id)}>
                        <Trash2 size={15} /> Xóa
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {!history.length && (
                <tr>
                  <td colSpan="4">Chưa có lịch sử.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function CapabilityStrip() {
  const items = [
    [FileVideo, "Upload video"],
    [Waypoints, "ROI trực quan"],
    [ShieldCheck, "State machine"],
    [Database, "SQLite history"]
  ];
  return (
    <section className="bg-ivory px-5 py-12 sm:px-8">
      <div className="mx-auto grid max-w-7xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {items.map(([Icon, label]) => (
          <Reveal key={label}>
            <div className="panel flex items-center gap-4 p-5">
              <Icon className="text-cocoa" size={22} />
              <span className="text-sm uppercase tracking-[0.22em] text-cocoa/70">{label}</span>
            </div>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [cameraId, setCameraId] = useState("");
  const [activeJobId, setActiveJobId] = useState("");
  const [activeFilename, setActiveFilename] = useState("");
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [history, setHistory] = useState([]);
  const [processingOptions, setProcessingOptions] = useState({ mode: "balanced", write_output_video: true });

  const stats = useMemo(
    () => ({
      videos: videos.filter((item) => item.source === "uploads").length,
      jobs: history.length,
      violations: history.reduce((sum, item) => sum + (item.violation_count || 0), 0)
    }),
    [videos, history]
  );

  async function loadVideos() {
    const data = await api("/api/videos");
    setVideos(data.videos || []);
  }

  async function loadHistory() {
    const data = await api("/api/history");
    setHistory(data.videos || []);
  }

  async function loadResults(jobId = activeJobId) {
    if (!jobId) return;
    const data = await api(`/api/results/${encodeURIComponent(jobId)}`);
    setResults(data.data);
    window.location.hash = "results";
  }

  async function processVideo(filename, options) {
    setActiveFilename(filename);
    const data = await api(`/api/process/${encodeURIComponent(filename)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options || processingOptions)
    });
    setActiveJobId(data.job_id);
    setStatus({ status: "starting", progress: 0, violations_found: 0 });
    window.location.hash = "processing";
  }

  useEffect(() => {
    loadVideos().catch(console.error);
    loadHistory().catch(console.error);
  }, []);

  useEffect(() => {
    if (!activeJobId) return undefined;
    const timer = window.setInterval(async () => {
      const data = await fetch(`/status/${encodeURIComponent(activeJobId)}`).then((res) => res.json());
      setStatus(data);
      if (data.status === "completed") {
        window.clearInterval(timer);
        await loadResults(activeJobId);
        await loadHistory();
      }
    }, 1800);
    return () => window.clearInterval(timer);
  }, [activeJobId]);

  return (
    <main className="min-h-screen bg-ivory text-ink">
      <Header activeJobId={activeJobId} />
      <Hero stats={stats} />
      <WaveDivider />
      <CapabilityStrip />
      <UploadSection
        onUploaded={(data) => {
          setVideos(data.videos || []);
          setSelectedVideo(data.path);
          setCameraId(data.camera_id);
        }}
      />
      <RoiSection
        videos={videos}
        selectedVideo={selectedVideo}
        setSelectedVideo={setSelectedVideo}
        cameraId={cameraId}
        setCameraId={setCameraId}
        processingOptions={processingOptions}
        setProcessingOptions={setProcessingOptions}
        onProcess={processVideo}
      />
      <ProcessingSection activeJobId={activeJobId} status={status} activeFilename={activeFilename} />
      <ResultsSection results={results} onRefresh={() => loadResults()} />
      <SearchSection />
      <HistorySection history={history} onLoadHistory={loadHistory} onOpenResults={loadResults} />
      <footer className="bg-ivory px-5 pb-10 text-center text-xs uppercase tracking-[0.28em] text-cocoa/60">
        RedLight AI Studio · YOLOv8 vehicle detection · YOLOv5 license plate recognition
      </footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
