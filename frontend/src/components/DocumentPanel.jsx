import { useState, useRef, useEffect } from "react";
import {
  Upload,
  X,
  FileText,
  File,
  Image,
  Loader2,
  Trash2,
  ChevronDown,
  Search,
  Download,
  RefreshCw,
  Eye,
} from "lucide-react";
import axios from "axios";

const DocumentPanel = ({ sessionId, isOpen, onClose, useRAG, onToggleRAG }) => {
  const [documents, setDocuments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [showChunks, setShowChunks] = useState(false);
  const [chunks, setChunks] = useState([]);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const fileInputRef = useRef(null);
  const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

  useEffect(() => {
    if (isOpen) {
      loadDocuments();
    }
  }, [isOpen]);

  const loadDocuments = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/api/documents`, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
      });
      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error("Error loading documents:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = async (event) => {
    const files = event.target.files;
    if (!files) return;

    for (const file of files) {
      await uploadDocument(file);
    }
    
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const uploadDocument = async (file) => {
    // Validate file type
    const supportedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain",
      "image/png",
      "image/jpeg",
      "image/jpg",
    ];

    if (!supportedTypes.includes(file.type)) {
      alert(`Unsupported file type: ${file.type}\nSupported: PDF, DOCX, TXT, PNG, JPG`);
      return;
    }

    // Check file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      alert("File too large. Maximum size: 50MB");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("title", file.name.split(".")[0]);

      const response = await axios.post(
        `${API_BASE}/api/documents/upload`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(percentCompleted);
          },
        }
      );

      // Reload documents list
      await loadDocuments();
      
      // Show success message
      const notification = document.createElement("div");
      notification.className =
        "fixed bottom-4 right-4 bg-green-500 text-white px-4 py-3 rounded-lg shadow-lg";
      notification.textContent = `✓ Uploaded: ${file.name} (${response.data.chunks} chunks)`;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    } catch (error) {
      console.error("Error uploading document:", error);
      const errorMsg = error.response?.data?.detail || error.message;
      const notification = document.createElement("div");
      notification.className =
        "fixed bottom-4 right-4 bg-red-500 text-white px-4 py-3 rounded-lg shadow-lg";
      notification.textContent = `✗ Error: ${errorMsg}`;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const deleteDocument = async (docId) => {
    if (!confirm("Are you sure you want to delete this document?")) return;

    try {
      await axios.delete(`${API_BASE}/api/documents/${docId}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
      });
      
      // Reload documents
      await loadDocuments();
      setSelectedDoc(null);

      const notification = document.createElement("div");
      notification.className =
        "fixed bottom-4 right-4 bg-green-500 text-white px-4 py-3 rounded-lg shadow-lg";
      notification.textContent = "Document deleted successfully";
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    } catch (error) {
      console.error("Error deleting document:", error);
      const notification = document.createElement("div");
      notification.className =
        "fixed bottom-4 right-4 bg-red-500 text-white px-4 py-3 rounded-lg shadow-lg";
      notification.textContent = `Error: ${error.response?.data?.detail || error.message}`;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    }
  };

  const regenerateEmbeddings = async (docId) => {
    if (!confirm("Regenerate embeddings for this document? This may take a moment."))
      return;

    try {
      setIsLoading(true);
      const response = await axios.post(
        `${API_BASE}/api/documents/${docId}/regenerate`,
        {},
        {
          headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
        }
      );

      // Reload documents
      await loadDocuments();

      const notification = document.createElement("div");
      notification.className =
        "fixed bottom-4 right-4 bg-green-500 text-white px-4 py-3 rounded-lg shadow-lg";
      notification.textContent = `Embeddings regenerated (${response.data.new_chunks} chunks)`;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    } catch (error) {
      console.error("Error regenerating embeddings:", error);
      const notification = document.createElement("div");
      notification.className =
        "fixed bottom-4 right-4 bg-red-500 text-white px-4 py-3 rounded-lg shadow-lg";
      notification.textContent = `Error: ${error.response?.data?.detail || error.message}`;
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    } finally {
      setIsLoading(false);
    }
  };

  const viewChunks = async (docId) => {
    setLoadingChunks(true);
    try {
      const response = await axios.get(`${API_BASE}/api/documents/${docId}/chunks`, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
      });
      setChunks(response.data.chunks || []);
      setShowChunks(true);
    } catch (error) {
      console.error("Error loading chunks:", error);
      const notification = document.createElement("div");
      notification.className =
        "fixed bottom-4 right-4 bg-red-500 text-white px-4 py-3 rounded-lg shadow-lg";
      notification.textContent = "Error loading chunks";
      document.body.appendChild(notification);
      setTimeout(() => notification.remove(), 3000);
    } finally {
      setLoadingChunks(false);
    }
  };

  const getFileIcon = (fileType) => {
    if (fileType.includes("pdf")) return <FileText className="w-4 h-4 text-red-500" />;
    if (fileType.includes("word")) return <FileText className="w-4 h-4 text-blue-500" />;
    if (fileType.includes("text")) return <FileText className="w-4 h-4 text-gray-500" />;
    if (fileType.includes("image")) return <Image className="w-4 h-4 text-purple-500" />;
    return <File className="w-4 h-4 text-gray-500" />;
  };

  const filteredDocs = documents.filter(
    (doc) =>
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6 flex justify-between items-center">
          <h2 className="text-xl font-bold">Document Manager</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-blue-500 rounded-lg transition"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {/* RAG Toggle */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-800">Use RAG Search</h3>
              <p className="text-sm text-gray-600">
                Enable to include uploaded documents in agent responses
              </p>
            </div>
            <button
              onClick={onToggleRAG}
              className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                useRAG ? "bg-green-500" : "bg-gray-300"
              }`}
            >
              <span
                className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                  useRAG ? "translate-x-7" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          {/* Upload Area */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center hover:bg-blue-50 cursor-pointer transition"
          >
            {isUploading ? (
              <div className="space-y-2">
                <Loader2 className="w-8 h-8 animate-spin mx-auto text-blue-600" />
                <p className="text-gray-600">Uploading: {uploadProgress}%</p>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="w-8 h-8 mx-auto text-blue-600" />
                <p className="text-gray-800 font-semibold">
                  Drop documents here or click to upload
                </p>
                <p className="text-sm text-gray-600">
                  Supports: PDF, DOCX, TXT, PNG, JPG (max 50MB)
                </p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileSelect}
              accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
              className="hidden"
              disabled={isUploading}
            />
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-3 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Documents List */}
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-800">
              Uploaded Documents ({filteredDocs.length})
            </h3>

            {isLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
              </div>
            ) : filteredDocs.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No documents uploaded yet</p>
              </div>
            ) : (
              <div className="space-y-2">
                {filteredDocs.map((doc) => (
                  <div
                    key={doc.doc_id}
                    className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3 flex-1">
                        {getFileIcon(doc.file_type)}
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-800">{doc.title}</h4>
                          <p className="text-sm text-gray-600">{doc.filename}</p>
                          <p className="text-xs text-gray-500 mt-1">
                            {doc.chunk_count} chunks • {new Date(doc.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-2 ml-4">
                        <button
                          onClick={() => {
                            setSelectedDoc(doc);
                            viewChunks(doc.doc_id);
                          }}
                          title="View chunks"
                          className="p-2 text-gray-500 hover:bg-blue-100 hover:text-blue-600 rounded-lg transition"
                        >
                          <Eye className="w-4 h-4" />
                        </button>

                        <button
                          onClick={() => regenerateEmbeddings(doc.doc_id)}
                          title="Regenerate embeddings"
                          className="p-2 text-gray-500 hover:bg-orange-100 hover:text-orange-600 rounded-lg transition"
                        >
                          <RefreshCw className="w-4 h-4" />
                        </button>

                        <button
                          onClick={() => deleteDocument(doc.doc_id)}
                          title="Delete document"
                          className="p-2 text-gray-500 hover:bg-red-100 hover:text-red-600 rounded-lg transition"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Chunks Modal */}
      {showChunks && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-gradient-to-r from-purple-600 to-purple-700 text-white p-6 flex justify-between items-center">
              <h2 className="text-xl font-bold">
                Document Chunks - {selectedDoc?.title} ({chunks.length} chunks)
              </h2>
              <button
                onClick={() => setShowChunks(false)}
                className="p-1 hover:bg-purple-500 rounded-lg transition"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              {loadingChunks ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-purple-600" />
                </div>
              ) : (
                chunks.map((chunk, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded-lg p-4 bg-gray-50 hover:bg-gray-100 transition"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-800">Chunk {chunk.index + 1}</h4>
                      <span className="text-xs text-gray-500">
                        {chunk.token_count} tokens
                      </span>
                    </div>
                    <p className="text-gray-700 text-sm whitespace-pre-wrap">
                      {chunk.text.substring(0, 500)}
                      {chunk.text.length > 500 && "..."}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentPanel;
