import { motion } from 'framer-motion';

interface KPIDashboardProps {
  activeShipments: number;
  onTimeRate: number;
  totalReroutes: number;
  totalDeliveries: number;
  timeSaved: number;
  avgDeliveryTime: number;
}

interface KPICardProps {
  label: string;
  value: string;
  color: string;
  delta?: string;
  deltaPositive?: boolean;
  index: number;
}

function KPICard({ label, value, color, delta, deltaPositive, index }: KPICardProps) {
  return (
    <motion.div
      className="kpi-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.4 }}
    >
      <span className="kpi-label">{label}</span>
      <span className={`kpi-value ${color}`}>{value}</span>
      {delta && (
        <span className={`kpi-delta ${deltaPositive ? 'positive' : 'negative'}`}>
          {deltaPositive ? '↑' : '↓'} {delta}
        </span>
      )}
    </motion.div>
  );
}

export default function KPIDashboard({
  activeShipments, onTimeRate, totalReroutes,
  totalDeliveries, timeSaved, avgDeliveryTime,
}: KPIDashboardProps) {
  return (
    <div className="kpi-grid">
      <KPICard index={0} label="Active Shipments" value={String(activeShipments)} color="blue" />
      <KPICard index={1} label="Deliveries" value={String(totalDeliveries)} color="emerald"
        delta={`${totalDeliveries} completed`} deltaPositive />
      <KPICard index={2} label="Reroutes" value={String(totalReroutes)} color="amber"
        delta="dynamic adjustments" deltaPositive />
      <KPICard index={3} label="Time Saved" value={`${Math.round(timeSaved)}s`} color="purple"
        delta="via predictions" deltaPositive />
      <KPICard index={4} label="Avg Delivery" value={`${Math.round(avgDeliveryTime)}s`} color="indigo" />
      <KPICard index={5} label="Efficiency"
        value={`${Math.min(99, Math.round(onTimeRate))}%`}
        color={onTimeRate > 80 ? 'emerald' : onTimeRate > 60 ? 'amber' : 'red'} />
    </div>
  );
}
