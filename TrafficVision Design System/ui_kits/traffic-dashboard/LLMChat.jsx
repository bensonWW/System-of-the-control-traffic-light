// LLMChat.jsx — AI assistant chat panel
const SAMPLE_MESSAGES = [
  { role: 'assistant', text: '您好！我是 TrafficVision AI 助理。請問有什麼關於目前交通數據或號誌控制的問題？', time: '11:20' },
  { role: 'user', text: '請問目前建國北路的車流狀況如何？', time: '11:22' },
  { role: 'assistant', text: '根據最新數據（11:22），**建國北路（長安東路–忠孝東路段）** 目前處於 🔴 壅塞（MOE 2）狀態。\n\n平均車速僅 **9.8 km/h**，佔有率高達 **47.7%**，車流量為 30 輛。\n\n建議考慮調整 IK7KP 號誌時制，延長主幹道綠燈時間以疏解壅塞。', time: '11:22', highlight: true },
];

function ChatMessage({ msg }) {
  const isUser = msg.role === 'user';
  const lines = msg.text.split('\n').filter(Boolean);
  return (
    <div style={{display:'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: 10}}>
      {!isUser && (
        <div style={chatStyles.aiAvatar}>AI</div>
      )}
      <div style={{maxWidth:'82%'}}>
        <div style={{...chatStyles.bubble, ...(isUser ? chatStyles.userBubble : chatStyles.aiBubble)}}>
          {lines.map((line, i) => {
            const parts = line.split(/\*\*(.*?)\*\*/g);
            return <p key={i} style={{margin: i > 0 ? '6px 0 0' : 0, lineHeight: 1.6}}>{parts.map((p, j) => j%2===1 ? <strong key={j} style={{color:'#e2e8f0',fontWeight:600}}>{p}</strong> : p)}</p>;
          })}
        </div>
        <div style={{...chatStyles.time, textAlign: isUser ? 'right' : 'left'}}>{msg.time}</div>
      </div>
    </div>
  );
}

function LLMChat() {
  const [messages, setMessages] = React.useState(SAMPLE_MESSAGES);
  const [input, setInput] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const bottomRef = React.useRef(null);

  const send = () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', text: input, time: new Date().toLocaleTimeString('zh-TW', {hour:'2-digit',minute:'2-digit'}) };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setLoading(true);
    setTimeout(() => {
      setMessages(m => [...m, { role: 'assistant', text: '正在分析交通數據，請稍候… 根據最新模擬結果，目前號誌優化建議已更新，請查看「號誌控制」面板。', time: new Date().toLocaleTimeString('zh-TW', {hour:'2-digit',minute:'2-digit'}) }]);
      setLoading(false);
    }, 1200);
  };

  return (
    <div style={chatStyles.wrap}>
      <div style={chatStyles.header}>
        <div style={chatStyles.headerLeft}>
          <div style={chatStyles.headerIcon}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          </div>
          <span style={chatStyles.headerTitle}>AI 數據助理</span>
        </div>
        <span style={chatStyles.modelTag}>claude-haiku</span>
      </div>
      <div style={chatStyles.messages}>
        {messages.map((m, i) => <ChatMessage key={i} msg={m}/>)}
        {loading && (
          <div style={{display:'flex',gap:6,padding:'4px 0'}}>
            <div style={chatStyles.aiAvatar}>AI</div>
            <div style={{...chatStyles.bubble, ...chatStyles.aiBubble, color:'#475569'}}>正在思考中…</div>
          </div>
        )}
        <div ref={bottomRef}/>
      </div>
      <div style={chatStyles.inputArea}>
        <div style={chatStyles.suggestions}>
          {['壅塞原因分析', '優化建議', '預測未來1小時'].map(s => (
            <button key={s} style={chatStyles.suggestion} onClick={() => setInput(s)}>{s}</button>
          ))}
        </div>
        <div style={chatStyles.inputRow}>
          <input
            style={chatStyles.input}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            placeholder="詢問交通狀況、號誌建議..."
          />
          <button style={chatStyles.sendBtn} onClick={send}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>
    </div>
  );
}

const chatStyles = {
  wrap: { display: 'flex', flexDirection: 'column', background: '#1a2236', border: '1px solid #2a3555', borderRadius: 8, overflow: 'hidden', height: '100%' },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', borderBottom: '1px solid #2a3555', flexShrink: 0 },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 8 },
  headerIcon: { width: 26, height: 26, borderRadius: 6, background: 'rgba(59,130,246,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' },
  headerTitle: { fontFamily: "'Space Grotesk',sans-serif", fontSize: 13, fontWeight: 600, color: '#e2e8f0' },
  modelTag: { fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: '#475569', background: '#111827', borderRadius: 4, padding: '2px 7px' },
  messages: { flex: 1, overflowY: 'auto', padding: '14px', display: 'flex', flexDirection: 'column' },
  aiAvatar: { width: 26, height: 26, borderRadius: 6, background: 'rgba(59,130,246,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: "'Space Grotesk',sans-serif", fontSize: 10, fontWeight: 700, color: '#60a5fa', flexShrink: 0, marginRight: 8, marginTop: 2 },
  bubble: { borderRadius: 10, padding: '9px 13px', fontSize: 12, lineHeight: 1.6, fontFamily: "'Noto Sans TC',sans-serif" },
  userBubble: { background: '#2563eb', color: '#fff', borderRadius: '10px 10px 3px 10px' },
  aiBubble: { background: '#111827', border: '1px solid #2a3555', color: '#94a3b8', borderRadius: '3px 10px 10px 10px' },
  time: { fontFamily: "'JetBrains Mono',monospace", fontSize: 10, color: '#334155', marginTop: 3 },
  inputArea: { padding: '10px 12px', borderTop: '1px solid #2a3555', flexShrink: 0 },
  suggestions: { display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' },
  suggestion: { fontFamily: "'Noto Sans TC',sans-serif", fontSize: 11, color: '#94a3b8', background: '#111827', border: '1px solid #2a3555', borderRadius: 4, padding: '3px 8px', cursor: 'pointer' },
  inputRow: { display: 'flex', gap: 8 },
  input: { flex: 1, background: '#111827', border: '1px solid #2a3555', borderRadius: 6, padding: '8px 12px', color: '#e2e8f0', fontFamily: "'Noto Sans TC',sans-serif", fontSize: 12, outline: 'none' },
  sendBtn: { width: 36, height: 36, background: '#3b82f6', border: 'none', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0 },
};

Object.assign(window, { LLMChat });
