import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE = 'https://logic-stics.onrender.com';
const WS_URL = 'wss://logic-stics.onrender.com/ws/live';

export interface SimState {
  step: number;
  traffic: {
    speeds: number[];
    time_of_day: number;
    day_of_week?: number;
    current_speed?: number;
    // MapView error fix karne ke liye disruptions add kiya
    disruptions?: { node_id: number; severity: number; remaining_steps: number; event_type: string }[];
  };
  prediction: {
    predicted_speeds: number[][];
    bottleneck_nodes: number[];
    bottleneck_severity: Record<string, number>;
    threshold?: number;
    mean_predicted_speed?: number;
  } | null;
  fleet: {
    vehicles: Vehicle[];
    active_count: number;
    total_deliveries: number;
    total_reroutes: number;
    total_time_saved: number;
    avg_delivery_time: number;
  };
  events: SimEvent[];
  bottleneck_nodes: number[];
  // Time Sync & Hybrid Engine fields
  live_anchor_speed?: number;
  is_live_synced?: boolean;
  speed_multiplier?: number;
  time_offset?: number;
}

export interface Vehicle {
  id: number;
  origin: number;
  destination: number;
  current_node: number;
  route: number[];
  route_index?: number;
  status: string;
  total_time?: number;
  total_distance?: number;
  reroute_count?: number;
  cargo_type?: string;
  progress: number;
}

export interface SimEvent {
  type: string;
  msg?: string;
  vehicle_id?: number;   // AlertFeed error fix
  time_saved?: number;   // AlertFeed error fix
  new_path?: number[];
  node_id?: number;
  severity?: number;     // AlertFeed error fix
  event_type?: string;
  step?: number;
  total_time?: number;   // AlertFeed error fix
}

export interface GraphData {
  nodes: { id: number; x: number; y: number; road_length?: number; speed_limit?: number; lanes?: number; road_class?: number }[];
  edges: { source: number; target: number }[];
  grid_size: number;
}

export function useSimulation() {
  const [state, setState] = useState<SimState | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // 1) Fetch Graph Topology once
  useEffect(() => {
    fetch(`${API_BASE}/api/graph`)
      .then(r => r.json())
      .then(data => {
        console.log("[useSimulation] Graph Data Loaded:", data);
        setGraph(data);
      })
      .catch(err => console.error("[useSimulation] Graph Fetch Error:", err));
  }, []);

  // 2) WebSocket connection with Auto-Reconnect
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket Connected ✅");
        setConnected(true);
      };

      ws.onclose = () => {
        console.log("WebSocket Disconnected ❌. Reconnecting...");
        setConnected(false);
        setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
        ws.close();
      };

      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data) as SimState;
          setState(data);
          if (data.events && data.events.length > 0) {
            setEvents(prev => [...data.events, ...prev].slice(0, 50));
          }
        } catch (e) {
          console.error("Parse Error:", e);
        }
      };
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  // 3) API Call: Inject Disruption
  const injectDisruption = useCallback(async (nodeId: number, severity = 0.05, eventType = 'accident') => {
    try {
      await fetch(`${API_BASE}/api/disruption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: nodeId, severity, radius: 3, duration: 24, event_type: eventType }),
      });
    } catch (err) {
      console.error("API Error (Disruption):", err);
    }
  }, []);

  // 4) API Call: Set Speed / Time Offset
  const setSpeed = useCallback(async (multiplier: number) => {
    try {
      await fetch(`${API_BASE}/api/speed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ multiplier }),
      });
    } catch (err) {
      console.error("API Error (Speed):", err);
    }
  }, []);

  return { state, graph, connected, events, injectDisruption, setSpeed };
}