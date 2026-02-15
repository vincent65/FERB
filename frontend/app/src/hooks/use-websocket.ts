"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { api } from "@/lib/api-client";
import type { WSEvent } from "@/lib/types";

interface UseWebSocketOptions {
  runId: string;
  enabled?: boolean;
  onEvent?: (event: WSEvent) => void;
}

export function useWebSocket({ runId, enabled = true, onEvent }: UseWebSocketOptions) {
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<WSEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!enabled || !runId) return;

    const url = api.getWSUrl(runId);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (messageEvent) => {
      try {
        const event: WSEvent = JSON.parse(messageEvent.data);
        setEvents((prev) => [...prev, event]);
        onEvent?.(event);
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [runId, enabled, onEvent]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { connected, events };
}
