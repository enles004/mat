"use client";

import { useEffect, useState } from "react";
import { API_URL, health, predict, type PredictResponse } from "./lib/api";
import styles from "./page.module.css";

const LABELS = ["negative", "neutral", "positive"] as const;

const STATUS_TEXT: Record<string, string> = {
  checking: "đang kiểm tra…",
  up: "đã kết nối",
  down: "không nối được — hãy `docker compose up` service ai trước",
};

const FILL_CLASS: Record<(typeof LABELS)[number], string> = {
  negative: styles.fillNegative,
  neutral: styles.fillNeutral,
  positive: styles.fillPositive,
};

export default function Home() {
  const [text, setText] = useState("Xe nhà Hyundai chạy ổn nhưng hơi hao xăng");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "up" | "down">("checking");

  useEffect(() => {
    health()
      .then(() => setApiStatus("up"))
      .catch(() => setApiStatus("down"));
  }, []);

  async function onPredict() {
    setLoading(true);
    setError(null);
    try {
      setResult(await predict(text));
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Lỗi không xác định");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <header>
          <p className={styles.eyebrow}>
            MAT <span className={styles.eyebrowAccent}>·</span> Vietnamese Automotive
            Sentiment
          </p>
          <h1 className={styles.title}>Phân loại cảm xúc bình luận ô tô</h1>
        </header>

        <p className={styles.status}>
          <span className={`${styles.statusDot} ${styles[apiStatus]}`} />
          API ({API_URL}): {STATUS_TEXT[apiStatus]}
        </p>

        <section className={styles.card}>
          <textarea
            className={styles.input}
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Nhập bình luận tiếng Việt…"
          />
          <div className={styles.cardFooter}>
            <span className={styles.hint}>Nhãn: negative / neutral / positive</span>
            <button
              className={styles.primary}
              onClick={onPredict}
              disabled={loading || text.trim().length === 0}
            >
              {loading ? "Đang chấm…" : "Chấm cảm xúc"}
            </button>
          </div>
        </section>

        {error !== null && <p className={`${styles.card} ${styles.error}`}>{error}</p>}

        {result !== null && (
          <section className={styles.card}>
            <div className={styles.labelRow}>
              <span className={`${styles.labelName} ${styles[result.label]}`}>
                {result.label}
              </span>
              {result.uncertain && <span className={styles.badge}>không chắc</span>}
            </div>
            <p className={styles.confidence}>
              Confidence
              <span className={styles.confidenceValue}>
                {result.confidence.toFixed(4)}
              </span>
            </p>
            <ul className={styles.scores}>
              {LABELS.map((label) => (
                <li key={label} className={styles.scoreRow}>
                  <span className={styles.scoreName}>{label}</span>
                  <div className={styles.bar}>
                    <div
                      className={`${styles.fill} ${FILL_CLASS[label]}`}
                      style={{ width: `${(result.scores[label] * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className={styles.scoreValue}>
                    {result.scores[label].toFixed(4)}
                  </span>
                </li>
              ))}
            </ul>
            <p className={styles.meta}>
              model: {result.model.backend} {result.model.version}
              {result.model.degraded ? " (degraded)" : ""} · request_id:{" "}
              {result.request_id}
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
