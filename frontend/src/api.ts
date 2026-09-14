// Thin wrappers around dashboard/server.py's REST command endpoints. Each
// call just forwards a JSON command over ZeroMQ to the sim process -- see
// that file and scripts/run_sim.py for what actually happens.

const API_BASE = import.meta.env.VITE_DASHBOARD_API_URL ?? "http://127.0.0.1:8000";

async function post(path: string, body?: unknown): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`);
  }
}

export const resetEpisode = (): Promise<void> => post("/api/command/reset");
export const pauseSim = (): Promise<void> => post("/api/command/pause");
export const resumeSim = (): Promise<void> => post("/api/command/resume");
export const stepSim = (): Promise<void> => post("/api/command/step");
export const setDomainRandomization = (enabled: boolean): Promise<void> =>
  post("/api/command/domain_randomization", { enabled });
