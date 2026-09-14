import { useState, useEffect } from 'react';
import type { PipelineSuspect } from '@/lib/api';
import api from '@/lib/api';

export function useSuspects(caseId: string | undefined) {
  const [suspects, setSuspects] = useState<PipelineSuspect[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!caseId) return;
    setLoading(true);
    setError(null);
    api
      .getPipelineSuspects(caseId)
      .then((data) => setSuspects(data.suspects))
      .catch((err) => {
        const status = (err as { response?: { status?: number } })?.response?.status;
        setError(
          status === 404
            ? 'Pipeline has not ranked suspects for this case yet.'
            : (err as Error).message,
        );
      })
      .finally(() => setLoading(false));
  }, [caseId]);

  return { suspects, loading, error };
}
