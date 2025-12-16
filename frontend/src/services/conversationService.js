/**
 * Conversation Service - API-Based
 * Manages conversation history using backend PostgreSQL database
 */

import axios from 'axios';

export const conversationService = {
  /**
   * Get all conversations from backend API
   */
  getConversations: async () => {
    try {
      const response = await axios.get('/api/sessions');
      const sessions = response.data.sessions || [];
      
      // Transform API response to frontend format
      return sessions.map(session => ({
        id: session.id,
        title: session.title || 'New Conversation',
        messages: [], // Messages loaded separately if needed
        createdAt: session.created_at,
        updatedAt: session.updated_at,
        starred: false, // Backend doesn't have star field yet
        use_rag: session.use_rag
      }));
    } catch (error) {
      console.error('Error fetching conversations from API:', error);
      return [];
    }
  },

  /**
   * Get a specific conversation by ID
   */
  getConversation: async (id) => {
    try {
      const response = await axios.get(`/api/sessions/${id}`);
      const session = response.data;
      
      return {
        id: session.id,
        title: session.title || 'New Conversation',
        messages: (session.messages || []).map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.created_at
        })),
        createdAt: session.created_at,
        updatedAt: session.updated_at,
        starred: false,
        use_rag: session.use_rag
      };
    } catch (error) {
      console.error('Error fetching conversation from API:', error);
      return null;
    }
  },

  /**
   * Create a new conversation
   */
  createConversation: async (title = 'New Conversation', messages = []) => {
    try {
      const response = await axios.post('/api/sessions', {
        title: title,
        use_rag: true
      });
      
      const session = response.data;
      return {
        id: session.id,
        title: session.title || 'New Conversation',
        messages: messages,
        createdAt: session.created_at,
        updatedAt: session.updated_at,
        starred: false,
        use_rag: session.use_rag
      };
    } catch (error) {
      console.error('Error creating conversation:', error);
      throw error;
    }
  },

  /**
   * Update conversation title
   */
  updateConversation: async (id, updates) => {
    try {
      if (updates.title) {
        const response = await axios.put(`/api/sessions/${id}?title=${encodeURIComponent(updates.title)}`);
        const session = response.data;
        
        return {
          id: session.id,
          title: session.title || 'New Conversation',
          messages: updates.messages || [],
          createdAt: session.created_at,
          updatedAt: session.updated_at,
          starred: updates.starred || false,
          use_rag: session.use_rag
        };
      }
      
      return null;
    } catch (error) {
      console.error('Error updating conversation:', error);
      throw error;
    }
  },

  /**
   * Save messages to a conversation (via chat API)
   * This is handled automatically by the /api/chat endpoint,
   * so we don't need to explicitly save here
   */
  saveMessages: async (id, messages) => {
    // Messages are saved automatically when using /api/chat endpoint
    // This is kept for compatibility with the frontend
    return true;
  },

  /**
   * Delete a conversation
   */
  deleteConversation: async (id) => {
    try {
      await axios.delete(`/api/sessions/${id}`);
      return await conversationService.getConversations();
    } catch (error) {
      console.error('Error deleting conversation:', error);
      throw error;
    }
  },

  /**
   * Delete empty conversations (no user messages)
   */
  deleteEmptyConversations: async () => {
    try {
      const conversations = await conversationService.getConversations();
      const toDelete = [];
      
      for (const conv of conversations) {
        const fullConv = await conversationService.getConversation(conv.id);
        if (!fullConv || !fullConv.messages || fullConv.messages.length === 0 || 
            (fullConv.messages.length === 1 && fullConv.messages[0].role === 'assistant')) {
          toDelete.push(conv.id);
        }
      }
      
      // Delete empty conversations
      for (const id of toDelete) {
        try {
          await conversationService.deleteConversation(id);
        } catch (e) {
          console.warn(`Failed to delete empty conversation ${id}:`, e);
        }
      }
      
      return toDelete.length;
    } catch (error) {
      console.error('Error in deleteEmptyConversations:', error);
      return 0;
    }
  },

  /**
   * Toggle star on conversation (local storage for now)
   */
  toggleStar: (id) => {
    // Note: Backend doesn't have star field yet
    // This is kept for frontend compatibility
    return null;
  },

  /**
   * Rename a conversation
   */
  renameConversation: async (id, newTitle) => {
    try {
      const response = await axios.put(`/api/sessions/${id}?title=${encodeURIComponent(newTitle)}`);
      const session = response.data;
      
      return {
        id: session.id,
        title: session.title,
        messages: [],
        createdAt: session.created_at,
        updatedAt: session.updated_at,
        starred: false,
        use_rag: session.use_rag
      };
    } catch (error) {
      console.error('Error renaming conversation:', error);
      throw error;
    }
  },

  /**
   * Clear all conversations (local only - not deleting from server)
   */
  clearAll: () => {
    return [];
  },

  /**
   * Export conversations as JSON
   */
  export: async () => {
    try {
      const conversations = await conversationService.getConversations();
      const dataStr = JSON.stringify(conversations, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
      
      const exportFileDefaultName = `conversations_${new Date().toISOString().split('T')[0]}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    } catch (error) {
      console.error('Error exporting conversations:', error);
    }
  }
};

export default conversationService;
