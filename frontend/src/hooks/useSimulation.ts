import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE = 'https://logic-stics.onrender.com';
const WS_URL = 'wss://logic-stics.onrender.com/ws/live';

// UPDATED INTERFACE: All new backend fields added
export interface SimState {
  step: number;
  traffic: {
    speeds: number[];
    time_of_day: number;
    day_of_week?: number;
    current_speed?: number;
  };
  prediction: {
    predicted_speeds: number[][];
    bottleneck_nodes: number[];
    bottleneck_severity: Record<string, number>;
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
  
  // ✅ CRITICAL: New Fields for Time Travel & Sync
  live_anchor_speed?: number;
  is_live_synced?: boolean;
  time_offset?: number; 
  speed_multiplier?: number;
}

export interface Vehicle {
  id: number;
  origin: number;
  destination: number;
  current_node: number;
  route: number[];
  status: string;
  progress: number;
}

export interface SimEvent {
  type: string;
  msg?: string;
  node_id?: number;
  event_type?: string;
  step?: number;
}

export interface GraphData {
  nodes: { id: number; x: number; y: number }[];
  edges: { source: number; target: number }[];
  grid_size: number;
}

export function useSimulation() {
  const [state, setState] = useState<SimState | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch graph topology
  useEffect(() => {
    fetch(`${API_BASE}/api/graph`)
      .then(r => r.json())
      .then(data => {
        console.log("Graph Loaded:", data);
        setGraph(data);
      })
      .catch(err => console.error("Graph Fetch Error:", err));
  }, []);

  // WebSocket connection
  useEffect(() => {
    const connect = () => {
      console.log("Connecting to WebSocket...");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WebSocket Connected ✅");
        setConnected(true);
      };

      ws.onclose = () => {
        console.log("WebSocket Disconnected ❌");
        setConnected(false);
        setTimeout(connect, 3000); // Retry
      };

      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          setState(data);
          if (data.events && data.events.length > 0) {
            setEvents(prev => [...data.events, ...prev].slice(0, 50));
          }
        } catch (e) {
          console.error("Data Parse Error:", e);
        }
      };
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  // FIXED: Inject Disruption
  const injectDisruption = useCallback(async (nodeId: number, severity = 0.05, eventType = 'accident') => {
    try {
      await fetch(`${API_BASE}/api/disruption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_id: nodeId, severity, radius: 3, duration: 24, event_type: eventType }),
      });
      console.log(`Disruption Sent to Node ${nodeId}`);
    } catch (err) {
      console.error("Disruption API Error:", err);
    }
  }, []);

  // FIXED: Time Travel (using multiplier as offset)
  const setSpeed = useCallback(async (multiplier: number) => {
    try {
      await fetch(`${API_BASE}/api/speed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ multiplier }),
      });
    } catch (err) {
      console.error("Speed API Error:", err);
    }
  }, []);

  return { state, graph, connected, events, injectDisruption, setSpeed };
}