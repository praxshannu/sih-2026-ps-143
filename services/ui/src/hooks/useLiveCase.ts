import { useEffect } from 'react';
import { useCasesStore } from '@/store/casesStore';
import { useWsStore } from '@/store/wsStore';

export function useLiveCase(caseId: string | undefined) {
  const fetchCase = useCasesStore((s) => s.fetchCase);
  const activeCase = useCasesStore((s) => s.activeCase);
  const loading = useCasesStore((s) => s.loading);
  const error = useCasesStore((s) => s.error);
  const connect = useWsStore((s) => s.connect);
  const connected = useWsStore((s) => s.connected);
  const lastMessage = useWsStore((s) => s.lastMessage);
  const progress = useWsStore((s) => s.progress);

  useEffect(() => {
    connect(caseId);
  }, [connect, caseId]);

  useEffect(() => {
    if (caseId) fetchCase(caseId);
  }, [caseId, fetchCase]);

  useEffect(() => {
    if (!caseId) return;
    if (
      lastMessage?.type === 'case_update' &&
      (lastMessage.payload as { id?: string })?.id === caseId
    ) {
      fetchCase(caseId);
    }
    if (
      lastMessage?.type === 'SUSPECT_RANKED' &&
      (lastMessage.payload as { case_id?: string })?.case_id === caseId
    ) {
      fetchCase(caseId);
    }
  }, [lastMessage, caseId, fetchCase]);

  return { activeCase, loading, error, connected, progress, lastMessage };
}
