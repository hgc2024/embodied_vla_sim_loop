import { useEffect, useRef, useState } from "react";
import type { DashboardMessage, FrameMessage, StatusMessage } from "./types";

const WS_URL = import.meta.env.VITE_DASHBOARD_WS_URL ?? "ws://127.0.0.1:8000/ws";
const RECONNECT_DELAY_MS = 1000;

export interface DashboardSocketState {
  connected: boolean;
  frame: FrameMessage | null;
  status: StatusMessage | null;
}

/** Owns the WebSocket connection to dashboard/server.py, with auto-reconnect
 * (the dashboard is expected to restart often during development -- see
 * sim/zmq_publisher.py's CommandPublisher docstring for the same reasoning
 * on the Python side). */
export function useDashboardSocket(): DashboardSocketState {
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<FrameMessage | null>(null);
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    function connect(): void {
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => setConnected(true);

      socket.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };

      socket.onerror = () => socket.close();

      socket.onmessage = (event: MessageEvent<string>) => {
        const message = JSON.parse(event.data) as DashboardMessage;
        if (message.type === "frame") {
          setFrame(message);
        } else if (message.type === "status") {
          setStatus(message);
        }
      };
    }

    connect();
    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer);
      socketRef.current?.close();
    };
  }, []);

  return { connected, frame, status };
}
