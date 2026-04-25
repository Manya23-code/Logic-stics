import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/live';

export interface SimState {
  step: number;
  traffic: {
    step: number;
    time_of_day: number;
    day_of_week: number;
    speeds: number[];
    disruptions: { node_id: number; severity: number; remaining_steps: number; event_type: string }[];
  };
  prediction: {
    predicted_speeds: number[][];
    bottleneck_nodes: number[];
    bottleneck_severity: Record<string, number>;
    threshold: number;
    mean_predicted_speed: number;
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
}

export interface Vehicle {
  id: number;
  origin: number;
  destination: number;
  current_node: number;
  route: number[];
  route_index: number;
  status: string;
  total_time: number;
  total_distance: number;
  reroute_count: number;
  cargo_type: string;
  progress: number;
}

export interface SimEvent {
  type: string;
  vehicle_id?: number;
  time_saved?: number;
  new_path?: number[];
  node_id?: number;
  severity?: number;
  event_type?: string;
  step?: number;
  total_time?: number;
}

export interface GraphData {
  nodes: { id: number; x: number; y: number; road_length: number; speed_limit: number; lanes: number; road_class: number }[];
  edges: { source: number; target: number }[];
  grid_size: number;
}

export function useSimulation() {
  const [state, setState] = useState<SimState | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch graph topology once
  useEffect(() => {
    fetch(`${API_BASE}/api/graph`)
      .then(r => r.json())
      .then(setGraph)
      .catch(console.error);
  }, []);

  // WebSocket connection
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data) as SimState;
          setState(data);
          if (data.events?.length > 0) {
            setEvents(prev => [...data.events, ...prev].slice(0, 100));
          }
        } catch {}
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  const injectDisruption = useCallback(async (nodeId: number, severity = 0.2, eventType = 'accident') => {
    await fetch(`${API_BASE}/api/disruption`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ node_id: nodeId, severity, radius: 3, duration: 24, event_type: eventType }),
    });
  }, []);

  const setSpeed = useCallback(async (multiplier: number) => {
    await fetch(`${API_BASE}/api/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ multiplier }),
    });
  }, []);

  const computeRoute = useCallback(async (origin: number, destination: number) => {
    const res = await fetch(`${API_BASE}/api/route`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin, destination }),
    });
    return res.json();
  }, []);

  return { state, graph, connected, events, injectDisruption, setSpeed, computeRoute };
}
