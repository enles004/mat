export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type SentimentLabel = "negative" | "neutral" | "positive";

export interface ModelInfo {
  backend: string;
  version: string;
  degraded: boolean;
}

export interface PredictResponse {
  label: SentimentLabel;
  confidence: number;
  scores: Record<SentimentLabel, number>;
  uncertain: boolean;
  model: ModelInfo;
  request_id: string;
}

export interface HealthResponse {
  status: string;
  model?: string;
  model_version?: string;
  [key: string]: unknown;
}

export interface ProblemDetails {
  title?: string;
  status?: number;
  detail?: string;
  code?: string;
  [key: string]: unknown;
}

export async function predict(text: string): Promise<PredictResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(await describeProblem(response));
  }
  return (await response.json()) as PredictResponse;
}

export async function health(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/health-check`);
  if (!response.ok) {
    throw new Error(await describeProblem(response));
  }
  return (await response.json()) as HealthResponse;
}

async function describeProblem(response: Response): Promise<string> {
  let detail = `HTTP ${response.status}`;
  try {
    const problem = (await response.json()) as ProblemDetails;
    detail = problem.detail ?? problem.title ?? detail;
    if (problem.code) detail = `[${problem.code}] ${detail}`;
  } catch {
    // Non-JSON error body; keep the HTTP status description.
  }
  return detail;
}
