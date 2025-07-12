document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const uploadLabel = document.querySelector('.upload-label');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const uploadSection = document.getElementById('uploadSection');
    const querySection = document.getElementById('querySection');
    const queryInput = document.getElementById('queryInput');
    const searchButton = document.getElementById('searchButton');
    const chatHistory = document.getElementById('chatHistory');
    const statusOverlay = document.getElementById('statusOverlay');
    const statusText = document.getElementById('statusText');
    const progressDetails = document.getElementById('progressDetails');
    const documentTree = document.getElementById('documentTree');
    
    let currentUserId = null;
    let eventSource = null;
    let currentTaskId = null;

    // Drag and drop functionality
    uploadLabel.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadLabel.classList.add('dragover');
    });

    uploadLabel.addEventListener('dragleave', () => {
        uploadLabel.classList.remove('dragover');
    });

    uploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadLabel.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileUpload();
        }
    });

    fileInput.addEventListener('change', handleFileUpload);

    function handleFileUpload() {
        const file = fileInput.files[0];
        if (!file) return;

        // Show progress UI
        uploadLabel.classList.add('hidden');
        progressContainer.classList.remove('hidden');
        
        // Simulate upload progress (in real app, use XHR with progress events)
        const totalSize = file.size;
        let uploaded = 0;
        const chunkSize = 1024 * 1024; // 1MB chunks
        
        const uploadInterval = setInterval(() => {
            uploaded += chunkSize;
            if (uploaded > totalSize) uploaded = totalSize;
            
            const percent = Math.min(100, (uploaded / totalSize) * 100);
            updateProgress(percent, uploaded, totalSize);
            
            if (percent >= 100) {
                clearInterval(uploadInterval);
                actuallyUploadFile(file);
            }
        }, 100);
    }

    function updateProgress(percent, uploaded, total) {
        progressBar.style.width = `${percent}%`;
        progressText.textContent = `${percent.toFixed(0)}%`;
        
        // Format sizes
        const uploadedMB = (uploaded / (1024 * 1024)).toFixed(1);
        const totalMB = (total / (1024 * 1024)).toFixed(1);
        progressText.textContent = `${uploadedMB}MB / ${totalMB}MB (${percent.toFixed(0)}%)`;
    }

    function actuallyUploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.task_id) {
                currentUserId = data.user_id;
                currentTaskId = data.task_id;
                showProcessingStatus();
                monitorTaskProgress(data.task_id);
            } else {
                showError(data.error || 'Upload failed');
            }
        })
        .catch(error => {
            showError(error.message);
        });
    }

    function showProcessingStatus() {
        statusOverlay.classList.remove('hidden');
        uploadSection.classList.add('hidden');
    }

    function monitorTaskProgress(taskId) {
        const progressInterval = setInterval(() => {
            fetch(`/status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    updateStatusUI(data);
                    
                    if (data.state === 'SUCCESS') {
                        clearInterval(progressInterval);
                        processingComplete();
                    } else if (data.state === 'FAILURE') {
                        clearInterval(progressInterval);
                        showError(data.status || 'Processing failed');
                    }
                })
                .catch(error => {
                    clearInterval(progressInterval);
                    showError('Error checking status');
                });
        }, 1000);
    }

    function updateStatusUI(data) {
        statusText.textContent = data.status || 'Processing...';
        
        if (data.current && data.total) {
            progressDetails.textContent = `Processing ${data.current} of ${data.total}`;
        } else {
            progressDetails.textContent = '';
        }
        
        if (data.document_tree && Object.keys(data.document_tree).length > 0) {
            documentTree.innerHTML = renderDocumentTree(data.document_tree);
        }
    }

    function renderDocumentTree(tree, indent = 0) {
        let html = '<ul>';
        for (const [key, value] of Object.entries(tree)) {
            html += `<li style="margin-left: ${indent * 15}px">
                <span class="file-icon">${value ? '📁' : '📄'}</span>
                <span class="file-name">${key}</span>`;
            if (value) {
                html += renderDocumentTree(value, indent + 1);
            }
            html += '</li>';
        }
        html += '</ul>';
        return html;
    }

    function processingComplete() {
        statusOverlay.classList.add('hidden');
        querySection.classList.remove('hidden');
        queryInput.focus();
        
        // Add system message to chat
        addMessage('system', 'Your documents are ready for analysis. Ask me anything about them.');
    }

    function showError(message) {
        statusText.textContent = `Error: ${message}`;
        statusOverlay.classList.add('error');
        
        // Reset UI after delay
        setTimeout(() => {
            statusOverlay.classList.add('hidden');
            statusOverlay.classList.remove('error');
            uploadLabel.classList.remove('hidden');
            progressContainer.classList.add('hidden');
        }, 3000);
    }

    // Query handling
    searchButton.addEventListener('click', handleQuery);
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleQuery();
        }
    });

    function handleQuery() {
        const query = queryInput.value.trim();
        if (!query || !currentUserId) return;
        
        // Add user message to chat
        addMessage('user', query);
        queryInput.value = '';
        
        // Create AI message element
        const aiMessageId = 'ai-' + Date.now();
        addMessage('ai', '', aiMessageId);
        
        // Scroll to bottom
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        if (eventSource) {
            eventSource.close();
        }
        
        eventSource = new EventSource(`/query?query=${encodeURIComponent(query)}&user_id=${currentUserId}`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'references') {
                // Store references for later use
                const refContainer = document.createElement('div');
                refContainer.className = 'references-container hidden';
                refContainer.id = `${aiMessageId}-refs`;
                
                data.content.forEach(ref => {
                    const refElement = document.createElement('div');
                    refElement.className = 'reference';
                    refElement.innerHTML = `
                        <span class="ref-number">[${ref.id}]</span>
                        <span class="ref-filename">${ref.filename}</span>
                        <div class="ref-excerpt">${ref.excerpt}</div>
                    `;
                    refElement.addEventListener('click', () => {
                        alert(`Would open document: ${ref.path}`);
                    });
                    refContainer.appendChild(refElement);
                });
                
                document.getElementById(aiMessageId).appendChild(refContainer);
            } 
            else if (data.type === 'text') {
                // Stream text to AI message
                const aiMessage = document.getElementById(aiMessageId);
                const contentElement = aiMessage.querySelector('.message-content') || 
                    document.createElement('div');
                
                contentElement.className = 'message-content';
                contentElement.innerHTML += data.content;
                aiMessage.appendChild(contentElement);
                
                // Scroll to bottom as content streams
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
            else if (!data.type) {
                // End of stream
                eventSource.close();
                
                // Show references
                const refContainer = document.getElementById(`${aiMessageId}-refs`);
                if (refContainer) {
                    refContainer.classList.remove('hidden');
                    
                    // Add "Sources" heading
                    const sourcesHeading = document.createElement('div');
                    sourcesHeading.className = 'sources-heading';
                    sourcesHeading.textContent = 'Sources';
                    refContainer.prepend(sourcesHeading);
                }
            }
        };
        
        eventSource.onerror = () => {
            document.getElementById(aiMessageId).innerHTML += 
                '<div class="error">Error generating response</div>';
            eventSource.close();
        };
    }

    function addMessage(role, content, id = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        if (id) messageDiv.id = id;
        
        if (role === 'user') {
            messageDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar user-avatar">👤</div>
                    <div class="message-role">You</div>
                </div>
                <div class="message-content">${content}</div>
            `;
        } 
        else if (role === 'ai') {
            messageDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar ai-avatar">🤖</div>
                    <div class="message-role">DocAnalyzer</div>
                </div>
            `;
            if (content) {
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = content;
                messageDiv.appendChild(contentDiv);
            }
        }
        else if (role === 'system') {
            messageDiv.innerHTML = `
                <div class="system-message">${content}</div>
            `;
        }
        
        chatHistory.appendChild(messageDiv);
        return messageDiv;
    }
});