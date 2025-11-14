import {
  User,
  Bot,
  Copy,
  Check,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";

const MessageBubble = ({ message }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Try to parse JSON content
  const tryParseJSON = (content) => {
    try {
      const parsed = JSON.parse(content);
      return { isJSON: true, data: parsed };
    } catch {
      return { isJSON: false, data: content };
    }
  };

  const { isJSON, data } = tryParseJSON(message.content);

  return (
    <div
      className={`flex items-start gap-3 animate-slide-in ${
        isUser ? "flex-row-reverse" : ""
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? "bg-gradient-to-br from-purple-500 to-pink-500"
            : "bg-gradient-to-br from-emerald-400 to-cyan-500"
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div
        className={`flex-1 max-w-3xl ${
          isUser ? "flex flex-col items-end" : ""
        }`}
      >
        <div
          className={`relative group ${
            isUser
              ? "bg-gradient-to-br from-purple-600 to-pink-600 text-white"
              : "bg-slate-800/50 text-slate-100 border border-slate-700"
          } rounded-2xl ${isUser ? "rounded-tr-none" : "rounded-tl-none"} p-4`}
        >
          {/* Copy button */}
          {!isUser && (
            <button
              onClick={() => copyToClipboard(message.content)}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-slate-700/50 hover:bg-slate-700 
                       opacity-0 group-hover:opacity-100 transition-opacity"
            >
              {copied ? (
                <Check className="w-4 h-4 text-emerald-400" />
              ) : (
                <Copy className="w-4 h-4 text-slate-400" />
              )}
            </button>
          )}

          {/* Content */}
          {isJSON ? (
            <JSONDisplay data={data} />
          ) : (
            <div className="prose prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  table: ({ children }) => (
                    <div className="my-4 overflow-x-auto">
                      <table className="min-w-full border-collapse bg-slate-900/50 rounded-lg overflow-hidden">
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children }) => (
                    <thead className="bg-gradient-to-r from-cyan-600 to-emerald-600">
                      {children}
                    </thead>
                  ),
                  tbody: ({ children }) => (
                    <tbody className="divide-y divide-slate-700">
                      {children}
                    </tbody>
                  ),
                  tr: ({ children, isHeader }) => (
                    <tr
                      className={
                        isHeader
                          ? ""
                          : "hover:bg-slate-800/50 transition-colors"
                      }
                    >
                      {children}
                    </tr>
                  ),
                  th: ({ children }) => (
                    <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase tracking-wider">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="px-4 py-3 text-sm text-slate-200 whitespace-nowrap">
                      {children}
                    </td>
                  ),
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        className="rounded-lg text-sm !my-3"
                        {...props}
                      >
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    ) : (
                      <code
                        className="bg-slate-700 px-1.5 py-0.5 rounded text-sm text-cyan-300"
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  p: ({ children }) => (
                    <p className="mb-3 last:mb-0 leading-relaxed text-slate-200">
                      {children}
                    </p>
                  ),
                  h1: ({ children }) => (
                    <h1 className="text-2xl font-bold text-white mb-3 mt-4 first:mt-0">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-xl font-bold text-white mb-2 mt-3 first:mt-0">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-lg font-semibold text-cyan-300 mb-2 mt-3 first:mt-0">
                      {children}
                    </h3>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc list-inside mb-3 space-y-1.5 text-slate-200">
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal list-inside mb-3 space-y-1.5 text-slate-200">
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => (
                    <li className="text-sm leading-relaxed">{children}</li>
                  ),
                  strong: ({ children }) => (
                    <strong className="font-semibold text-cyan-300">
                      {children}
                    </strong>
                  ),
                  em: ({ children }) => (
                    <em className="text-emerald-300 not-italic font-medium">
                      {children}
                    </em>
                  ),
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-cyan-500 pl-4 py-2 my-3 bg-slate-800/30 italic text-slate-300">
                      {children}
                    </blockquote>
                  ),
                  hr: () => <hr className="border-slate-700 my-4" />,
                }}
              >
                {data}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Component to display JSON data in a beautiful way
const JSONDisplay = ({ data }) => {
  const [expanded, setExpanded] = useState(true);

  if (typeof data !== "object" || data === null) {
    return <div className="text-sm font-mono">{String(data)}</div>;
  }

  return (
    <div className="space-y-2">
      {data.success !== undefined && (
        <div
          className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
            data.success
              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
              : "bg-red-500/20 text-red-400 border border-red-500/30"
          }`}
        >
          <div
            className={`w-2 h-2 rounded-full ${
              data.success ? "bg-emerald-400" : "bg-red-400"
            }`}
          />
          {data.success ? "Success" : "Failed"}
        </div>
      )}

      {data.message && (
        <p className="text-slate-300 text-sm mb-3">{data.message}</p>
      )}

      <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
        <pre className="text-xs overflow-x-auto">
          <code className="text-cyan-300">{JSON.stringify(data, null, 2)}</code>
        </pre>
      </div>
    </div>
  );
};

export default MessageBubble;
