// LLMChat.jsx — TrafficVision v2 · AI Data Assistant

/* ── v2 tokens ─────────────────────────────────── */
const _LC = {
  ff:   "'Inter', system-ui, sans-serif",
  body: "'Noto Sans TC', 'PingFang TC', sans-serif",
  mono: "'JetBrains Mono', monospace",
  bgCard:    '#0f1629',
  bgSurf:    '#0a0f1e',
  bgElev:    '#162036',
  borderMed: 'rgba(99,130,200,0.25)',
  divider:   'rgba(99,130,200,0.08)',
  shadowCard:'0 0 0 1px rgba(99,130,200,0.25), 0 1px 3px rgba(0,0,0,0.40)',
  blue400:   '#60a5fa',
  blue500:   '#3b82f6',
  blue600:   '#2563eb',
  blueSubtle:'rgba(59,130,246,0.10)',
  blueBd:    'rgba(59,130,246,0.20)',
  fg:  '#f1f5f9',
  fg2: '#94a3b8',
  fg3: '#4e6080',
  fgD: '#2d3d55',
};

const SAMPLE_MESSAGES = [
  { role: 'assistant', text: '您好！我是 TrafficVision AI 助理。請問有什麼關於目前交通數據或號誌控制的問題？', time: '11:20' },
  { role: 'user',      text: '請問目前建國北路的車流狀況如何？', time: '11:22' },
  { role: 'assistant', text: '根據最新數據（11:22），**建國北路（長安東路–忠孝東路段）** 目前處於 壅塞（MOE 2）狀態。\n\n平均車速僅 **9.8 km/h**，佔有率高達 **47.7%**，車流量為 30 輛。\n\n建議考慮調整 IK7KP 號誌時制，延長主幹道綠燈時間以疏解壅塞。', time: '11:22' },
];

function parseMarkdown(text) {
  return text.split(/\*\*(.*?)\*\*/g).map((part, i) =>
    i % 2 === 1
      ? <strong key={i} style={{ color: _LC.fg, fontWeight: 600 }}>{part}</strong>
      : part
  );
}

function ChatMessage({ msg }) {
  const L = _LC;
  const isUser = msg.role === 'user';
  const lines = msg.text.split('\n').filter(Boolean);

  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: 12, gap: 8 }}>
      {!isUser && (
        <div style={{
          width: 26, height: 26, borderRadius: 7,
          background: L.blueSubtle, border: `1px solid ${L.blueBd}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: L.ff, fontSize: 10, fontWeight: 700, color: L.blue400,
          flexShrink: 0, marginTop: 2,
        }}>AI</div>
      )}
      <div style={{ maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: 3, alignItems: isUser ? 'flex-end' : 'flex-start' }}>
        <div style={{
          borderRadius: isUser ? '10px 10px 3px 10px' : '3px 10px 10px 10px',
          padding: '9px 13px',
          background: isUser ? L.blue500 : L.bgElev,
          border: isUser ? 'none' : `1px solid ${L.borderMed}`,
          fontFamily: L.body, fontSize: 12, lineHeight: 1.65,
          color: isUser ? '#fff' : L.fg2,
        }}>
          {lines.map((line, i) => (
            <p key={i} style={{ margin: i > 0 ? '6px 0 0' : 0 }}>{parseMarkdown(line)}</p>
          ))}
        </div>
        <div style={{ fontFamily: L.mono, fontSize: 10, color: L.fgD }}>{msg.time}</div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 12 }}>
      <div style={{ width: 26, height: 26, borderRadius: 7, background: _LC.blueSubtle, border: `1px solid ${_LC.blueBd}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: _LC.ff, fontSize: 10, fontWeight: 700, color: _LC.blue400, flexShrink: 0 }}>AI</div>
      <div style={{ background: _LC.bgElev, border: `1px solid ${_LC.borderMed}`, borderRadius: '3px 10px 10px 10px', padding: '10px 14px', display: 'flex', gap: 5, alignItems: 'center' }}>
        {[0,1,2].map(i => (
          <div key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: _LC.fg3, animation: `bounce 1.2s ${i*0.2}s ease-in-out infinite` }}/>
        ))}
      </div>
    </div>
  );
}

const SUGGESTIONS = ['壅塞原因分析', '號誌優化建議', '預測未來1小時'];

function LLMChat() {
  const [messages, setMessages] = React.useState(SAMPLE_MESSAGES);
  const [input, setInput] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const bottomRef = React.useRef(null);
  const L = _LC;

  const now = () => new Date().toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });

  const send = () => {
    if (!input.trim() || loading) return;
    const userMsg = { role: 'user', text: input, time: now() };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setLoading(true);
    setTimeout(() => {
      setMessages(m => [...m, {
        role: 'assistant',
        text: '正在分析交通數據，請稍候… 根據最新模擬結果，目前號誌優化建議已更新，請查看「號誌控制」面板。',
        time: now(),
      }]);
      setLoading(false);
    }, 1200);
  };

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      background: L.bgCard, border: `1px solid ${L.borderMed}`,
      borderRadius: 12, boxShadow: L.shadowCard,
      overflow: 'hidden', height: '100%',
    }}>
      {/* ── Header ── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '11px 14px', borderBottom: `1px solid ${L.borderMed}`, flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 26, height: 26, borderRadius: 7, background: L.blueSubtle, border: `1px solid ${L.blueBd}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={L.blue400} strokeWidth="2" strokeLinecap="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          </div>
          <span style={{ fontFamily: L.ff, fontSize: 13, fontWeight: 600, color: L.fg, letterSpacing: '-0.01em' }}>AI 數據助理</span>
        </div>
        <span style={{ fontFamily: L.mono, fontSize: 10, color: L.fg3, background: L.bgSurf, border: `1px solid ${L.borderMed}`, borderRadius: 4, padding: '2px 7px' }}>claude-haiku</span>
      </div>

      {/* ── Messages ── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px', display: 'flex', flexDirection: 'column' }}>
        {messages.map((m, i) => <ChatMessage key={i} msg={m}/>)}
        {loading && <TypingIndicator/>}
        <div ref={bottomRef}/>
      </div>

      {/* ── Input area ── */}
      <div style={{ padding: '10px 12px', borderTop: `1px solid ${L.borderMed}`, flexShrink: 0 }}>
        {/* Suggestion chips */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
          {SUGGESTIONS.map(s => (
            <button key={s} onClick={() => setInput(s)} style={{
              fontFamily: L.body, fontSize: 11, color: L.fg2,
              background: L.bgSurf, border: `1px solid ${L.borderMed}`,
              borderRadius: 5, padding: '3px 9px', cursor: 'pointer',
              transition: 'border-color 150ms, color 150ms',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(99,130,200,0.45)'; e.currentTarget.style.color = L.fg; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = L.borderMed; e.currentTarget.style.color = L.fg2; }}
            >{s}</button>
          ))}
        </div>

        {/* Input row */}
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
            placeholder="詢問交通狀況、號誌建議..."
            style={{
              flex: 1, height: 36,
              background: L.bgSurf, border: `1px solid ${L.borderMed}`,
              borderRadius: 8, padding: '0 12px',
              color: L.fg, fontFamily: L.body, fontSize: 12,
              outline: 'none', transition: 'border-color 150ms',
            }}
            onFocus={e => e.target.style.borderColor = L.blue500}
            onBlur={e => e.target.style.borderColor = L.borderMed}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            style={{
              width: 36, height: 36, borderRadius: 8, border: 'none',
              background: (loading || !input.trim()) ? 'rgba(59,130,246,0.3)' : L.blue500,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: (loading || !input.trim()) ? 'not-allowed' : 'pointer',
              flexShrink: 0, transition: 'background 150ms',
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>

    </div>
  );
}

Object.assign(window, { LLMChat });
