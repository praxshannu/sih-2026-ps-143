import { useState, useEffect } from 'react';
import type { PipelineDrift } from '@/lib/api';
import api from '@/lib/api';

export function useDriftForecast(caseId: string | undefined) {
  const [forecast, setForecast] = useState<PipelineDrift | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    api
      .getPipelineDrift(caseId)
      .then((data) => setForecast(data))
      .catch((err) => {
        const status = (err as { response?: { status?: number } })?.response?.status;
        setError(
          status === 404
            ? 'Drift has not run for this case yet.'
            : (err as Error).message,
        );
      })
      .finally(() => setLoading(false));
  }, [caseId]);

  return { forecast, loading, error };
}
