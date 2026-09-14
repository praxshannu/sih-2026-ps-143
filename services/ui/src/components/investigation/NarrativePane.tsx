import { motion } from 'framer-motion';
import { FileText } from 'lucide-react';

interface NarrativePaneProps {
  narrative: string;
  title?: string;
}

export default function NarrativePane({ narrative, title = 'Case Summary' }: NarrativePaneProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel p-4"
    >
      <div className="mb-3 flex items-center gap-2">
        <FileText size={14} className="text-sentinel-primary" />
        <h3 className="font-display text-sm font-semibold text-sentinel-text">{title}</h3>
        <span className="ml-auto rounded bg-sentinel-primary/10 px-1.5 py-0.5 text-[9px] font-mono text-sentinel-primary">
          LLM GENERATED
        </span>
      </div>
      <div className="space-y-2 text-sm leading-relaxed text-sentinel-text/80">
        {narrative.split('\n').map((paragraph, i) => (
          <p key={i}>{paragraph}</p>
        ))}
      </div>
    </motion.div>
  );
}
