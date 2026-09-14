import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
} from 'recharts';
import type { Suspect } from '@/lib/api';
import { COLORS } from '@/lib/constants';

interface ScoreRadarProps {
  scores: Suspect['scores'];
}

const LABEL_MAP: Record<string, string> = {
  proximity: 'PROX',
  ais_gap: 'AIS',
  vessel_type: 'TYPE',
  speed_anomaly: 'SPD',
  course_deviation: 'CRS',
  historical: 'HIST',
};

export default function ScoreRadar({ scores }: ScoreRadarProps) {
  const data = Object.entries(scores).map(([key, value]) => ({
    factor: LABEL_MAP[key] ?? key,
    score: value as number,
    fullMark: 100,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke={COLORS.border} strokeDasharray="3 3" />
        <PolarAngleAxis
          dataKey="factor"
          tick={{ fill: COLORS.muted, fontSize: 9, fontFamily: 'JetBrains Mono' }}
        />
        <Radar
          dataKey="score"
          stroke={COLORS.danger}
          fill={COLORS.danger}
          fillOpacity={0.15}
          strokeWidth={1.5}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
