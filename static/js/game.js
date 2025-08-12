// グローバルな状態管理オブジェクト
const gameState = {
    currentQuestionCount: 0,
    maxQuestions: 5,
    selectedCandidates: [],
    questionTarget: 'all',
    currentConversationView: 'all',
    conversations: {
        all: []
    },
    candidates: [],
    suspiciousCandidate: null,
    company: {},
    isInitialized: false,
    userSelection: null
};

// APIエンドポイント
const API_URL = '/api/generate';
const STREAM_API_URL = '/api/generate_stream';
const DEBUG_API_URL = '/api/debug_logs';

// デバッグエリア関連の変数
let isUserScrolling = false;
let scrollTimeout;

// デバッグ情報を表示する関数（改善版）
function updateDebugDisplay() {
    fetch(DEBUG_API_URL)
        .then(response => response.json())
        .then(logs => {
            const debugContent = document.getElementById('debugContent');
            if (debugContent && logs.length > 0) {
                const wasAtBottom = debugContent.scrollTop >= debugContent.scrollHeight - debugContent.clientHeight - 10;
                
                debugContent.innerHTML = logs.map(log => `
                    <div class="debug-log ${log.type.toLowerCase()}">
                        <div class="debug-timestamp">[${log.timestamp}] ${log.type}</div>
                        <div class="debug-content">${log.content.replace(/\n/g, '<br>')}</div>
                    </div>
                `).join('');
                
                if (!isUserScrolling && wasAtBottom) {
                    debugContent.scrollTop = debugContent.scrollHeight;
                }
            }
        })
        .catch(error => console.error('Debug logs fetch failed:', error));
}

// スクロール検知とリサイズ機能を追加
function initializeDebugArea() {
    const debugContent = document.getElementById('debugContent');
    const debugArea = document.querySelector('.debug-area');
    const mainContainer = document.querySelector('.main-container');
    
    if (debugContent) {
        debugContent.addEventListener('scroll', () => {
            isUserScrolling = true;
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                isUserScrolling = false;
            }, 1000);
        });
    }
    
    if (debugArea && mainContainer) {
        createResizeHandle(debugArea, mainContainer);
    }
}

// リサイズハンドルを作成
function createResizeHandle(debugArea, mainContainer) {
    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'debug-resize-handle';
    resizeHandle.innerHTML = '⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯ デバッグエリアのサイズを調整 ⋯⋯⋯⋯⋯⋯⋯⋯⋯⋯';
    
    debugArea.insertBefore(resizeHandle, debugArea.firstChild);
    
    let isResizing = false;
    let startY = 0;
    let startHeight = 0;
    
    resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startHeight = parseInt(window.getComputedStyle(debugArea).height, 10);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        e.preventDefault();
    });
    
    function handleMouseMove(e) {
        if (!isResizing) return;
        
        const deltaY = startY - e.clientY;
        const newDebugHeight = Math.max(100, Math.min(window.innerHeight * 0.7, startHeight + deltaY));
        const newMainHeight = window.innerHeight - 80 - newDebugHeight;
        
        debugArea.style.height = `${newDebugHeight}px`;
        mainContainer.style.height = `${newMainHeight}px`;
    }
    
    function handleMouseUp() {
        isResizing = false;
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }
}

// ストリーミング対応のLLama API呼び出し（全会話履歴対応）
async function sendQuestionToLLamaAPIStream(question, candidate, conversationHistory, onChunk, onComplete, onError) {
    try {
        const response = await fetch(STREAM_API_URL, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'text/plain'
            },
            body: JSON.stringify({
                question: question,
                candidate: candidate,
                company: gameState.company,
                conversation_history: conversationHistory,
                all_conversations: gameState.conversations
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        // let buffer = '';
        // let streamingText = '';
        
        // while (true) {
        //     const { done, value } = await reader.read();
            
        //     if (done) break;
            
        //     buffer += decoder.decode(value, { stream: true });
            
        //     const lines = buffer.split('\n');
        //     buffer = lines.pop();
            
        //     for (const line of lines) {
        //         if (line.startsWith('data: ')) {
        //             try {
        //                 const data = JSON.parse(line.slice(6));
                        
        //                 if (data.status === 'generating' && data.chunk) {
        //                     streamingText += data.chunk;
        //                     onChunk(data.chunk, streamingText);
        //                 } else if (data.status === 'completed' && data.complete_response) {
        //                     onComplete(data.complete_response);
        //                     return;
        //                 } else if (data.status === 'error') {
        //                     onError(new Error(data.error || 'ストリーミングエラー'));
        //                     return;
        //                 }
        //             } catch (e) {
        //                 console.warn('Failed to parse streaming data:', line);
        //             }
        //         }
        //     }
        // }
        let buffer = '';
let streamingText = '';
let pendingRender = false;

function render() {
    // onChunkを呼び出してUIを更新
    onChunk(null, streamingText);
    pendingRender = false;
}

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
        if (line.startsWith('data: ')) {
            try {
                const data = JSON.parse(line.slice(6));
                if (data.status === 'generating' && data.chunk) {
                    streamingText += data.chunk;
                    // すぐにレンダリングするのではなく、フレームの更新を要求
                    if (!pendingRender) {
                        pendingRender = true;
                        requestAnimationFrame(render);
                    }
                } else if (data.status === 'completed' && data.complete_response) {
                    onComplete(data.complete_response);
                    return;
                } // ... (エラー処理など)
            } catch (e) {
                console.warn('Failed to parse streaming data:', line);
            }
        }
    }
}
        
    } catch (error) {
        onError(error);
    }
}

// 候補者選択画面を表示
function showCandidateSelection() {
    const modal = document.getElementById('selectionModal');
    const candidateOptions = document.getElementById('candidateOptions');
    
    candidateOptions.innerHTML = gameState.candidates.map((candidate, index) => `
        <div class="candidate-selection-card" data-index="${index}">
            <div class="candidate-info">
                <h4>${candidate.name}</h4>
                <p><strong>大学:</strong> ${candidate.university}</p>
                <p><strong>強み:</strong> ${candidate.strength}</p>
            </div>
            <div class="conversation-summary">
                <h5>💬 これまでの会話記録:</h5>
                <div class="answer-summary">
                    ${generateConversationRecord(candidate, index)}
                </div>
            </div>
            <button class="select-candidate-btn" onclick="selectSuspiciousCandidate(${index})">
                この人が最も志望度が低いと判断
            </button>
        </div>
    `).join('');
    
    modal.classList.add('active');
}

// 会話記録を生成（修正版：古い順に全て表示）
function generateConversationRecord(candidate, candidateIndex) {
    const messages = gameState.conversations[candidate.name] || [];
    const candidateAnswers = messages.filter(msg => msg.sender === candidate.name);
    
    if (candidateAnswers.length === 0) {
        return '<p class="no-answers">この候補者は質問に答えていません</p>';
    }
    
    return candidateAnswers.map((answer, index) => `
        <div class="answer-item">
            <strong>質問${index + 1}への回答:</strong>
            <p>"${answer.text}"</p>
            <small class="timestamp">回答時刻: ${new Date(answer.timestamp).toLocaleTimeString()}</small>
        </div>
    `).join('');
}

// ユーザーの候補者選択処理
function selectSuspiciousCandidate(selectedIndex) {
    gameState.userSelection = selectedIndex;
    
    document.getElementById('selectionModal').classList.remove('active');
    showFinalResult();
}

// 最終結果表示
function showFinalResult() {
    const modal = document.getElementById('resultModal');
    
    // 正解判定: ユーザーの選択と最も志望度の低い候補者（middle）が一致するか
    const isCorrect = gameState.userSelection === gameState.suspiciousCandidate;
    const score = isCorrect ? 100 : 0;
    
    // 結果表示
    document.getElementById('score').textContent = score + '点';
    document.getElementById('feedback').innerHTML = generateSubtleFeedback(isCorrect);
    
    modal.classList.add('active');
}

// フィードバック生成
function generateSubtleFeedback(isCorrect) {
    const correctCandidate = gameState.candidates[gameState.suspiciousCandidate];
    const userChoice = gameState.candidates[gameState.userSelection];
    
    let feedback = '';
    
    if (isCorrect) {
        feedback = `
            <div class="result-correct">
                <h3>🎉 正解です！</h3>
                <p><strong>${correctCandidate.name}</strong>が最も企業研究が不足していた候補者でした。</p>
                <p>微妙な差を見抜く素晴らしい観察眼です！</p>
            </div>
        `;
    } else {
        feedback = `
            <div class="result-incorrect">
                <h3>❌ 残念！</h3>
                <p>あなたの選択: <strong>${userChoice.name}</strong></p>
                <p>正解: <strong>${correctCandidate.name}</strong></p>
                <p>このレベルの差は非常に難しいです。次回はより注意深く観察してみましょう。</p>
            </div>
        `;
    }
    
    // 全候補者の実際の企業研究レベルを表示
    feedback += `
        <hr>
        <div class="preparation-levels">
            <h4>📊 実際の企業研究レベル</h4>
            ${gameState.candidates.map((candidate, index) => {
                const levelText = {
                    'high': '完璧（100%）🔥',
                    'high-middle': '高レベル（85%）📋', 
                    'middle': '中レベル（70%）📝'
                }[candidate.preparation];
                
                const isCorrectAnswer = index === gameState.suspiciousCandidate;
                const isUserChoice = index === gameState.userSelection;
                
                return `
                    <div class="candidate-level ${isCorrectAnswer ? 'correct-answer' : ''} ${isUserChoice ? 'user-choice' : ''}">
                        <strong>${candidate.name}</strong>: 企業研究レベル ${levelText}
                    </div>
                `;
            }).join('')}
        </div>
        <div class="game-difficulty-note">
            <p><small>※ このゲームでは全員が高い志望度を示すため、情報の精度や具体性のわずかな差で判断する必要があります。</small></p>
        </div>
    `;
    
    return feedback;
}

// HTMLの読み込み完了後にすべての処理を開始
document.addEventListener('DOMContentLoaded', function() {
    resetGameState();

    // DOM要素の取得
    const createButton = document.getElementById('createGameButton');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const questionInput = document.getElementById('questionInput');
    const askButton = document.getElementById('askButton');
    const evaluateButton = document.getElementById('evaluateButton');
    const restartButton = document.getElementById('restartButton');
    const closeResultButton = document.getElementById('closeResultButton');
    const targetSelector = document.querySelector('.target-selector');
    const candidatesList = document.getElementById('candidatesList');
    const conversationTabs = document.getElementById('conversationTabs');

    // イベントリスナーの設定
    if (createButton) {
        createButton.addEventListener('click', createGame);
    }
    
    if (askButton) {
        askButton.addEventListener('click', askQuestionStream);
    }

    if (evaluateButton) {
        evaluateButton.addEventListener('click', showEvaluation);
    }
    
    if (restartButton) {
        restartButton.addEventListener('click', () => location.reload());
    }

    if (closeResultButton) {
        closeResultButton.addEventListener('click', () => {
            document.getElementById('resultModal').classList.remove('active');
        });
    }

    if (questionInput) {
        questionInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !askButton.disabled) {
                askQuestionStream();
            }
        });
    }

    if (targetSelector) {
        targetSelector.addEventListener('click', (e) => {
            const targetOption = e.target.closest('.target-option');
            if (targetOption) {
                selectTarget(targetOption.dataset.target);
            }
        });
    }

    if (candidatesList) {
        candidatesList.addEventListener('click', (e) => {
            const card = e.target.closest('.candidate-card');
            if (card) {
                toggleCandidateSelection(parseInt(card.dataset.index));
            }
        });
    }

    if (conversationTabs) {
        conversationTabs.addEventListener('click', (e) => {
            const tab = e.target.closest('.conversation-tab');
            if(tab) {
                showConversation(tab.dataset.name);
            }
        });
    }

    setInterval(updateDebugDisplay, 3000);

    setTimeout(() => {
        initializeDebugArea();
    }, 1000);

    // 関数定義
    function resetGameState() {
        gameState.currentQuestionCount = 0;
        gameState.maxQuestions = 5;
        gameState.selectedCandidates = [];
        gameState.questionTarget = 'all';
        gameState.currentConversationView = 'all';
        gameState.conversations = { all: [] };
        gameState.candidates = [];
        gameState.suspiciousCandidate = null;
        gameState.company = {};
        gameState.isInitialized = false;
        gameState.userSelection = null;
        
        const questionCount = document.getElementById('questionCount');
        if (questionCount) {
            questionCount.textContent = '5';
        }
        
        const evaluateButton = document.getElementById('evaluateButton');
        if (evaluateButton) {
            evaluateButton.disabled = true;
        }
        
        console.log('🔄 ゲーム状態をリセットしました');
    }

    function createGame() {
        loadingOverlay.classList.remove('hidden');
        checkProgress(); 

        fetch('/api/create_game', { method: 'POST' })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.message || 'サーバーでエラーが発生しました。') });
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    initializeGameUI(data);
                } else {
                    throw new Error(data.message || 'ゲームデータの生成に失敗しました。');
                }
            })
            .catch(error => {
                console.error('Game creation failed:', error);
                const loadingMessage = document.getElementById('loadingMessage');
                if(loadingMessage) {
                    loadingMessage.textContent = 'エラー: ' + error.message;
                    loadingMessage.style.color = '#f5576c';
                }
                const progressFill = document.getElementById('progressFill');
                if(progressFill) {
                    progressFill.style.width = '100%';
                    progressFill.style.background = '#f5576c';
                }
                setTimeout(() => loadingOverlay.classList.add('hidden'), 5000);
            });
    }

    function checkProgress() {
        fetch('/api/progress')
            .then(response => response.json())
            .then(data => {
                const loadingMessage = document.getElementById('loadingMessage');
                const progressFill = document.getElementById('progressFill');
                const loadingProgress = document.getElementById('loadingProgress');
                
                if (loadingMessage) loadingMessage.textContent = data.message;
                if (progressFill) progressFill.style.width = data.progress + '%';
                if (loadingProgress) loadingProgress.textContent = data.progress + '%';
                
                if (data.status !== 'completed' && data.status !== 'error') {
                    setTimeout(checkProgress, 1000);
                }
            })
            .catch(error => {
                console.error('Progress check failed:', error);
            });
    }
    
    function initializeGameUI(data) {
        // 候補者をランダムにシャッフル
        data.candidates = data.candidates
            .slice() // 元の配列をコピー
            .sort(() => Math.random() - 0.5);

        // 企業情報を描画
        const company = data.company;
        const companyDetails = document.getElementById('companyDetails');
        if(companyDetails) {
            companyDetails.innerHTML = `
                <div class="company-detail"><strong>会社名：</strong> ${company.name || 'N/A'}</div>
                <div class="company-detail"><strong>事業内容：</strong> ${company.business || 'N/A'}</div>
                <div class="company-detail"><strong>売上高：</strong> ${company.revenue || 'N/A'}</div>
                <div class="company-detail"><strong>従業員数：</strong> ${company.employees || 'N/A'}</div>
                <div class="company-detail"><strong>設立：</strong> ${company.founded || 'N/A'}</div>
                <div class="company-detail"><strong>本社：</strong> ${company.location || 'N/A'}</div>
                <div class="company-detail"><strong>ビジョン：</strong> ${company.vision || 'N/A'}</div>
                <div class="company-detail"><strong>主力製品：</strong> ${company.products || 'N/A'}</div>
                <div class="company-detail"><strong>社風：</strong> ${company.culture || 'N/A'}</div>
                <div class="company-detail"><strong>最近のニュース：</strong> ${company.recent_news || 'N/A'}</div>
                <div class="company-detail"><strong>競合優位性：</strong> ${company.competitive_advantage || 'N/A'}</div>
                <div class="company-detail"><strong>CEO・代表メッセージ：</strong> ${company.ceo_message || 'N/A'}</div>
                <div class="company-detail"><strong>事業展開計画：</strong> ${company.expansion_plan || 'N/A'}</div>
                <div class="company-detail"><strong>受賞歴・評価：</strong> ${company.awards || 'N/A'}</div>
                <div class="company-detail"><strong>パートナーシップ・提携：</strong> ${company.partnerships || 'N/A'}</div>
            `;
        }
        
        // 候補者リストを描画（興味分野は非表示）
        const candidates = data.candidates;
        candidatesList.innerHTML = '';
        candidates.forEach((candidate, index) => {
            const cardHTML = `
                <div class="candidate-card" data-index="${index}">
                    <h4>${candidate.name || '不明な候補者'}</h4>
                    <p><strong>大学：</strong>${candidate.university || 'N/A'}</p>
                    <p><strong>ガクチカ：</strong>${candidate.gakuchika || 'N/A'}</p>
                    <p><strong>強み：</strong>${candidate.strength || 'N/A'}</p>
                    <!-- <p><strong>MBTI：</strong>${candidate.mbti || 'N/A'}</p> -->
                </div>
            `;
            candidatesList.insertAdjacentHTML('beforeend', cardHTML);
        });

        // 会話タブを描画
        conversationTabs.innerHTML = `<button class="conversation-tab active" data-name="all">全体の会話</button>`;
        candidates.forEach(candidate => {
            const tabHTML = `<button class="conversation-tab" data-name="${candidate.name}">${candidate.name}</button>`;
            conversationTabs.insertAdjacentHTML('beforeend', tabHTML);
        });

        initGame(data);

        document.getElementById('setupScreen').style.display = 'none';
        document.querySelector('.main-container').style.display = 'flex';
        
        loadingOverlay.classList.add('hidden');
    }

    function initGame(data) {
        resetGameState();
        
        Object.assign(gameState, {
            currentQuestionCount: 0,
            maxQuestions: 5,
            selectedCandidates: [],
            questionTarget: 'all',
            currentConversationView: 'all',
            conversations: { all: [] },
            candidates: data.candidates,
            company: data.company,
            isInitialized: true,
            userSelection: null
        });
        
        // 最も企業研究が不足している候補者（middle）を正解として設定
        gameState.suspiciousCandidate = gameState.candidates.findIndex(
            candidate => candidate.preparation === 'middle'
        );
        
        if (gameState.suspiciousCandidate === -1) {
            gameState.suspiciousCandidate = 0;
            console.warn('No middle preparation candidate found, using first candidate as fallback');
        }
        
        gameState.candidates.forEach(candidate => {
            gameState.conversations[candidate.name] = [];
        });
        
        updateQuestionCounter();
        
        console.log('🎯ゲーム初期化完了');
        console.log(`正解の候補者: ${gameState.candidates[gameState.suspiciousCandidate].name} (preparation: ${gameState.candidates[gameState.suspiciousCandidate].preparation})`);
        console.log(`企業研究レベル分布: ${gameState.candidates.map(c => c.preparation).join(', ')}`);
    }

    function updateQuestionCounter() {
        const questionCount = document.getElementById('questionCount');
        if (questionCount) {
            const remaining = gameState.maxQuestions - gameState.currentQuestionCount;
            questionCount.textContent = remaining.toString();
        }
    }

    function selectTarget(target) {
        gameState.questionTarget = target;
        document.querySelectorAll('.target-option').forEach(option => {
            option.classList.toggle('active', option.dataset.target === target);
        });
        
        const cards = document.querySelectorAll('.candidate-card');
        if (target === 'individual') {
            cards.forEach(card => {
                card.style.cursor = 'pointer';
                card.title = 'クリックして選択';
            });
        } else {
            gameState.selectedCandidates = [];
            cards.forEach(card => {
                card.classList.remove('selected');
                card.style.cursor = 'default';
                card.title = '';
            });
        }
    }

    function toggleCandidateSelection(index) {
        if (gameState.questionTarget !== 'individual') return;
        
        const card = document.querySelector(`[data-index="${index}"]`);
        const isSelected = gameState.selectedCandidates.includes(index);
        
        document.querySelectorAll('.candidate-card').forEach(c => c.classList.remove('selected'));
        
        if (isSelected) {
            gameState.selectedCandidates = [];
        } else {
            gameState.selectedCandidates = [index];
            if(card) card.classList.add('selected');
        }
    }

    // 質問の実行
    async function askQuestionStream() {
        const question = questionInput.value.trim();
        if (!question) { alert('質問を入力してください'); return; }
        if (gameState.currentQuestionCount >= gameState.maxQuestions) { alert('質問回数の上限に達しました'); return; }
        if (gameState.questionTarget === 'individual' && gameState.selectedCandidates.length === 0) { alert('質問する候補者を選択してください'); return; }
        
        askButton.disabled = true;
        askButton.innerHTML = '<span class="loading"></span>LLamaが回答生成中...';
        
        try {
            const targetCandidates = gameState.questionTarget === 'all' 
                ? gameState.candidates.map((_, i) => i)
                : gameState.selectedCandidates;
            
            const question_msg = { sender: 'interviewer', text: question, timestamp: new Date() };
            gameState.conversations.all.push(question_msg);
            targetCandidates.forEach(candidateIndex => {
                const candidate = gameState.candidates[candidateIndex];
                gameState.conversations[candidate.name].push(question_msg);
            });
            
            updateConversationDisplay();
            
            for (let i = 0; i < targetCandidates.length; i++) {
                const candidateIndex = targetCandidates[i];
                const candidate = gameState.candidates[candidateIndex];
                
                askButton.innerHTML = `<span class="loading"></span>${candidate.name}が回答中...`;
                
                const candidateMessage = { 
                    sender: candidate.name, 
                    text: '', 
                    timestamp: new Date(), 
                    preparation: candidate.preparation,
                    isStreaming: true
                };
                
                gameState.conversations.all.push(candidateMessage);
                gameState.conversations[candidate.name].push(candidateMessage);
                
                updateConversationDisplay();
                
                await new Promise((resolve, reject) => {
                    const fullConversationHistory = gameState.conversations[candidate.name];
                    
                    sendQuestionToLLamaAPIStream(
                        question, 
                        candidate, 
                        fullConversationHistory,
                        (chunk, cumulativeText) => {
                            // candidateMessage.text = cumulativeText || candidateMessage.text + chunk;
                            // updateConversationDisplay();
                            
                            // const content = document.getElementById('conversationContent');
                            // content.scrollTop = content.scrollHeight;

                            candidateMessage.text = cumulativeText; // テキストは更新しておく
                    
                            const streamingTextElement = document.getElementById('streaming-message-text');
                            if (streamingTextElement) {
                                // HTML全体を再描画するのではなく、テキスト部分だけを更新する
                                const escapedText = cumulativeText.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
                                streamingTextElement.innerHTML = `${escapedText}<span class="typing-cursor">▋</span>`;
                            
                                // スクロール処理はここでも行う
                                const content = document.getElementById('conversationContent');
                                content.scrollTop = content.scrollHeight;
                            }
                        },
                        (completeResponse) => {
                            candidateMessage.text = completeResponse;
                            candidateMessage.isStreaming = false;
                            updateConversationDisplay();
                            
                            const content = document.getElementById('conversationContent');
                            content.scrollTop = content.scrollHeight;
                            
                            resolve();
                        },
                        (error) => {
                            console.error(`${candidate.name}の回答生成エラー:`, error);
                            candidateMessage.text = "申し訳ございません、少し考えさせてください。";
                            candidateMessage.isStreaming = false;
                            updateConversationDisplay();
                            resolve();
                        }
                    );
                });
            }
            
            gameState.currentQuestionCount++;
            updateQuestionCounter();
            questionInput.value = '';
            
            setTimeout(updateDebugDisplay, 1000);
            
            if (gameState.questionTarget === 'individual') {
                gameState.selectedCandidates = [];
                document.querySelectorAll('.candidate-card').forEach(card => card.classList.remove('selected'));
            }
            
            if (gameState.currentQuestionCount >= 1) {
                evaluateButton.disabled = false;
            }
            
        } catch (error) {
            alert('質問処理中にエラーが発生しました。');
            console.error('質問送信エラー:', error);
        } finally {
            askButton.disabled = false;
            askButton.innerHTML = '質問する';
        }
    }

    function showEvaluation() {
        if (gameState.currentQuestionCount < gameState.maxQuestions) {
            if (!confirm(`まだ${gameState.maxQuestions - gameState.currentQuestionCount}回質問できます。本当に評価に進みますか？`)) {
                return;
            }
        }
        
        showCandidateSelection();
    }

    function showConversation(target) {
        gameState.currentConversationView = target;
        document.querySelectorAll('.conversation-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.name === target);
        });
        updateConversationDisplay();
    }

    // function updateConversationDisplay() {
    //     const content = document.getElementById('conversationContent');
    //     const messages = gameState.conversations[gameState.currentConversationView] || [];
        
    //     if (messages.length === 0) {
    //         content.innerHTML = '<div class="no-conversation">まだ会話がありません</div>';
    //         return;
    //     }
        
    //     content.innerHTML = messages.map(msg => {
    //         const isInterviewer = msg.sender === 'interviewer';
    //         let senderLabel = isInterviewer ? '面接官' : msg.sender;
            
    //         const cursor = msg.isStreaming ? '<span class="typing-cursor">▋</span>' : '';
    //         const streamingClass = msg.isStreaming ? ' streaming' : '';
            
    //         const escapedText = msg.text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            
    //         return `
    //             <div class="message ${isInterviewer ? 'interviewer' : 'candidate'}${streamingClass}">
    //                 <div class="message-sender">${senderLabel}</div>
    //                 <div class="message-bubble">
    //                     <div class="message-text">${escapedText}${cursor}</div>
    //                 </div>
    //             </div>`;
    //     }).join('');
        
    //     content.scrollTop = content.scrollHeight;
    // }

    function updateConversationDisplay() {
        const content = document.getElementById('conversationContent');
        const messages = gameState.conversations[gameState.currentConversationView] || [];
    
        if (messages.length === 0) {
            content.innerHTML = '<div class="no-conversation">まだ会話がありません</div>';
            return;
        }
    
        content.innerHTML = messages.map(msg => {
            const isInterviewer = msg.sender === 'interviewer';
            let senderLabel = isInterviewer ? '面接官' : msg.sender;
        
            const cursor = msg.isStreaming ? '<span class="typing-cursor">▋</span>' : '';
            const streamingClass = msg.isStreaming ? ' streaming' : '';
            // ストリーミングメッセージ用のIDを設定
            const textElementId = msg.isStreaming ? 'id="streaming-message-text"' : '';
        
            const escapedText = msg.text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
            return `
                <div class="message ${isInterviewer ? 'interviewer' : 'candidate'}${streamingClass}">
                    <div class="message-sender">${senderLabel}</div>
                    <div class="message-bubble">
                        <div class="message-text" ${textElementId}>${escapedText}${cursor}</div>
                    </div>
                </div>`;
        }).join('');
    
        content.scrollTop = content.scrollHeight;
    }
});

// グローバル関数として定義（HTMLから呼び出されるため）
window.selectSuspiciousCandidate = selectSuspiciousCandidate;