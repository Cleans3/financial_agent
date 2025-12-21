import React, { useState, useEffect } from 'react'
import { Upload, Trash2, Eye, Search, FileText } from 'lucide-react'
import axios from 'axios'

const DocumentManagementPanel = ({ api }) => {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadingFile, setUploadingFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [totalDocs, setTotalDocs] = useState(0)
  const [currentPage, setCurrentPage] = useState(0)
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [deleteConfirm, setDeleteConfirm] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [uploadForm, setUploadForm] = useState({
    title: '',
    category: '',
    tags: ''
  })

  const LIMIT = 20

  useEffect(() => {
    fetchDocuments()
  }, [currentPage])

  const fetchDocuments = async () => {
    try {
      setLoading(true)
      const res = await api.get('/admin/documents', {
        params: { skip: currentPage * LIMIT, limit: LIMIT }
      })
      setDocuments(res.data.documents)
      setTotalDocs(res.data.total)
      setError(null)
    } catch (err) {
      setError('Failed to fetch documents')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0])
    }
  }

  const uploadFile = async (file) => {
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)
    formData.append('title', uploadForm.title || file.name)
    if (uploadForm.category) formData.append('category', uploadForm.category)
    if (uploadForm.tags) formData.append('tags', JSON.stringify(uploadForm.tags.split(',')))

    try {
      setUploadingFile(file.name)
      const res = await api.post('/admin/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      
      alert(`✓ Upload successful: ${res.data.chunks} chunks`)
      setUploadForm({ title: '', category: '', tags: '' })
      setCurrentPage(0)
      await fetchDocuments()
    } catch (err) {
      setError(`Upload failed: ${err.response?.data?.detail || err.message}`)
      console.error(err)
    } finally {
      setUploadingFile(null)
    }
  }

  const deleteDocument = async (docId) => {
    try {
      setLoading(true)
      await api.delete(`/admin/documents/${docId}`)
      alert('Document deleted')
      setDeleteConfirm(null)
      setCurrentPage(0)
      await fetchDocuments()
    } catch (err) {
      setError(`Delete failed: ${err.response?.data?.detail || err.message}`)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const viewDocumentDetails = async (docId) => {
    try {
      setLoading(true)
      const res = await api.get(`/admin/documents/${docId}`)
      setSelectedDoc(res.data)
    } catch (err) {
      setError('Failed to fetch document details')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const filteredDocs = documents.filter(doc =>
    doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doc.uploaded_by.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <div className="bg-gray-700 rounded-lg p-6 border border-gray-600">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white">
          <Upload size={20} /> Upload Document
        </h3>
        
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            dragActive ? 'border-blue-400 bg-blue-900 bg-opacity-30' : 'border-gray-500'
          }`}
        >
          <input
            type="file"
            id="file-upload"
            hidden
            onChange={(e) => e.target.files && uploadFile(e.target.files[0])}
          />
          <label htmlFor="file-upload" className="cursor-pointer block">
            <div className="text-gray-300">
              {uploadingFile ? (
                <p>Uploading: {uploadingFile}...</p>
              ) : (
                <>
                  <p className="font-medium">Drag & drop your file here</p>
                  <p className="text-sm">or click to browse</p>
                  <p className="text-xs text-gray-400 mt-2">PDF, DOCX, TXT, PNG, JPG (max 500MB)</p>
                </>
              )}
            </div>
          </label>
        </div>

        <div className="grid grid-cols-3 gap-4 mt-4">
          <input
            type="text"
            placeholder="Title (optional)"
            value={uploadForm.title}
            onChange={(e) => setUploadForm({...uploadForm, title: e.target.value})}
            className="bg-gray-600 border border-gray-500 rounded px-3 py-2 text-sm text-white placeholder-gray-400"
          />
          <input
            type="text"
            placeholder="Category (optional)"
            value={uploadForm.category}
            onChange={(e) => setUploadForm({...uploadForm, category: e.target.value})}
            className="bg-gray-600 border border-gray-500 rounded px-3 py-2 text-sm text-white placeholder-gray-400"
          />
          <input
            type="text"
            placeholder="Tags (comma-separated, optional)"
            value={uploadForm.tags}
            onChange={(e) => setUploadForm({...uploadForm, tags: e.target.value})}
            className="bg-gray-600 border border-gray-500 rounded px-3 py-2 text-sm text-white placeholder-gray-400"
          />
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-900 border border-red-700 rounded p-4 text-red-200 text-sm">
          {error}
        </div>
      )}

      {/* Documents List */}
      <div className="bg-gray-700 rounded-lg p-6 border border-gray-600">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2 text-white">
            <FileText size={20} /> Documents ({totalDocs} total)
          </h3>
          <input
            type="text"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-gray-600 border border-gray-500 rounded px-3 py-2 text-sm w-48 text-white placeholder-gray-400"
          />
        </div>

        {loading ? (
          <div className="text-center py-8 text-gray-400">Loading...</div>
        ) : filteredDocs.length === 0 ? (
          <div className="text-center py-8 text-gray-400">No documents uploaded yet</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-600">
                <tr>
                  <th className="px-4 py-3 text-left text-gray-200">Filename</th>
                  <th className="px-4 py-3 text-left text-gray-200">Uploaded By</th>
                  <th className="px-4 py-3 text-left text-gray-200">Date</th>
                  <th className="px-4 py-3 text-left text-gray-200">Size</th>
                  <th className="px-4 py-3 text-left text-gray-200">Chunks</th>
                  <th className="px-4 py-3 text-left text-gray-200">Category</th>
                  <th className="px-4 py-3 text-center text-gray-200">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredDocs.map((doc) => (
                  <tr key={doc.doc_id} className="border-t border-gray-600 hover:bg-gray-600">
                    <td className="px-4 py-3 truncate text-gray-100">{doc.filename}</td>
                    <td className="px-4 py-3 text-gray-300">{doc.uploaded_by}</td>
                    <td className="px-4 py-3 text-xs text-gray-400">
                      {new Date(doc.uploaded_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-3 text-xs text-gray-300">{(doc.file_size / 1024 / 1024).toFixed(2)} MB</td>
                    <td className="px-4 py-3 text-xs text-gray-300">{doc.chunks}</td>
                    <td className="px-4 py-3 text-xs text-gray-400">{doc.category || '-'}</td>
                    <td className="px-4 py-3 text-center">
                      <div className="flex justify-center gap-2">
                        <button
                          onClick={() => viewDocumentDetails(doc.doc_id)}
                          className="text-blue-400 hover:text-blue-300 p-1"
                          title="View details"
                        >
                          <Eye size={16} />
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(doc.doc_id)}
                          className="text-red-400 hover:text-red-300 p-1"
                          title="Delete"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {totalDocs > LIMIT && (
          <div className="flex justify-between items-center mt-4 pt-4 border-t border-gray-600">
            <button
              onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
              disabled={currentPage === 0}
              className="px-4 py-2 border border-gray-500 rounded disabled:opacity-50 text-sm text-gray-300 hover:bg-gray-600"
            >
              Previous
            </button>
            <span className="text-sm text-gray-400">
              Page {currentPage + 1} of {Math.ceil(totalDocs / LIMIT)}
            </span>
            <button
              onClick={() => setCurrentPage(p => p + 1)}
              disabled={(currentPage + 1) * LIMIT >= totalDocs}
              className="px-4 py-2 border border-gray-500 rounded disabled:opacity-50 text-sm text-gray-300 hover:bg-gray-600"
            >
              Next
            </button>
          </div>
        )}
      </div>

      {/* Delete Confirmation */}
      {deleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-sm border border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-white">Delete Document?</h3>
            <p className="text-gray-300 mb-6">This action cannot be undone. The document will be permanently removed from the vectorDB.</p>
            <div className="flex gap-4">
              <button
                onClick={() => setDeleteConfirm(null)}
                className="flex-1 px-4 py-2 border border-gray-600 rounded hover:bg-gray-700 text-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteDocument(deleteConfirm)}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Document Details Modal */}
      {selectedDoc && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-2xl max-h-[80vh] overflow-y-auto w-full mx-4 border border-gray-700">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">{selectedDoc.filename}</h3>
              <button
                onClick={() => setSelectedDoc(null)}
                className="text-gray-400 hover:text-gray-200"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="font-semibold text-gray-400">Uploaded By</p>
                  <p className="text-gray-200">{selectedDoc.upload_info?.uploaded_by}</p>
                </div>
                <div>
                  <p className="font-semibold text-gray-400">Date</p>
                  <p className="text-gray-200">{new Date(selectedDoc.upload_info?.uploaded_at).toLocaleString()}</p>
                </div>
                <div>
                  <p className="font-semibold text-gray-400">File Size</p>
                  <p className="text-gray-200">{(selectedDoc.upload_info?.file_size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <div>
                  <p className="font-semibold text-gray-400">Total Chunks</p>
                  <p className="text-gray-200">{selectedDoc.chunk_count}</p>
                </div>
              </div>

              {selectedDoc.upload_info?.category && (
                <div>
                  <p className="font-semibold text-gray-400 text-sm">Category</p>
                  <p className="text-sm text-gray-300">{selectedDoc.upload_info.category}</p>
                </div>
              )}

              {selectedDoc.upload_info?.tags?.length > 0 && (
                <div>
                  <p className="font-semibold text-gray-400 text-sm">Tags</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedDoc.upload_info.tags.map((tag, i) => (
                      <span key={i} className="bg-blue-900 text-blue-200 px-2 py-1 rounded text-xs">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <p className="font-semibold text-gray-400 text-sm mb-2">First 10 Chunks Preview</p>
                <div className="bg-gray-900 rounded p-4 space-y-3 text-xs border border-gray-600">
                  {selectedDoc.chunks?.slice(0, 10).map((chunk, i) => (
                    <div key={i} className="border-b border-gray-700 pb-2">
                      <p className="font-semibold text-gray-300">Chunk {i + 1}</p>
                      <p className="text-gray-400 line-clamp-3">{chunk.text || chunk}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex gap-4 mt-6">
              <button
                onClick={() => setSelectedDoc(null)}
                className="flex-1 px-4 py-2 border border-gray-600 rounded hover:bg-gray-700 text-gray-300"
              >
                Close
              </button>
              <button
                onClick={() => {
                  deleteDocument(selectedDoc.doc_id)
                  setSelectedDoc(null)
                }}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DocumentManagementPanel
