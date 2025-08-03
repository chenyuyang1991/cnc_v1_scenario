class ThreeDModelViewer extends HTMLElement {
  constructor() {
    super();
    this.isActive = false;
    this.renderEngine = null;
    this.sceneContainer = null;
    this.cameraController = null;
    this._shadowRoot = null;
    this.modelMesh = null;
    this.stats = null; // 新增：Stats 引用
    this.loadingOverlay = null; // 新增：載入覆蓋層
    this.progressBar = null; // 新增：進度條
    this.progressText = null; // 新增：進度文字
    this.loadingStatus = null; // 新增：載入狀態
  }

  connectedCallback() {
    this.isActive = true;
    console.log('[ThreeDModelViewer] connectedCallback 開始');
    try {
      this.initializeViewer();
    } catch (error) {
      console.error('[ThreeDModelViewer] 初始化失敗:', error);
      this.handleError(`初始化失敗: ${error.message}`);
    }
  }

  disconnectedCallback() {
    this.isActive = false;
    this.cleanup();
  }

  initializeViewer() {
    console.log('[ThreeDModelViewer] 開始初始化檢視器');
    
    // 創建 Shadow DOM 容器
    try {
      if (!this._shadowRoot) {
        this._shadowRoot = this.attachShadow({ mode: 'open' });
        console.log('[ThreeDModelViewer] Shadow DOM 創建成功');
      }
      
      const viewportDiv = document.createElement('div');
      this.applyContainerStyles(viewportDiv);
      this._shadowRoot.appendChild(viewportDiv);
      console.log('[ThreeDModelViewer] 容器 div 創建成功');
      
      // 創建載入畫面
      this.createLoadingOverlay();
      
    } catch (error) {
      console.error('[ThreeDModelViewer] Shadow DOM 創建失敗:', error);
      this.handleError(`Shadow DOM 創建失敗: ${error.message}`);
      return;
    }

    // 驗證必要屬性
    if (!this.hasAttribute('model')) {
      console.error('[ThreeDModelViewer] 缺少 model 屬性');
      this.handleError('必須提供 model 屬性');
      return;
    }

    // 解析屬性參數
    const config = this.parseAttributes();
    console.log('[ThreeDModelViewer] 配置解析完成:', config);
    
    // 初始化 3D 環境
    try {
      const viewportDiv = this._shadowRoot.querySelector('div');
      this.setup3DEnvironment(viewportDiv, config);
      console.log('[ThreeDModelViewer] 3D 環境設置完成');
    } catch (error) {
      console.error('[ThreeDModelViewer] 3D 環境設置失敗:', error);
      this.handleError(`3D 環境設置失敗: ${error.message}`);
      return;
    }
    
    // 載入 3D 模型
    try {
      this.loadModel(config);
      console.log('[ThreeDModelViewer] 開始載入模型');
    } catch (error) {
      console.error('[ThreeDModelViewer] 模型載入失敗:', error);
      this.handleError(`模型載入失敗: ${error.message}`);
    }
  }

  parseAttributes() {
    const config = {
      modelPath: this.getAttribute('model'),
      modelColor: parseInt(this.getAttribute('color')?.replace("#", "0x"), 16),
      autoRotation: this.getAttribute('auto_rotate') === 'true',
      materialOpacity: parseFloat(this.getAttribute('opacity')),
      surfaceShininess: Number(this.getAttribute('shininess')),
      renderMode: this.getAttribute('materialType'),
      verticalAngle: Number(this.getAttribute('cam_v_angle')),
      horizontalAngle: Number(this.getAttribute('cam_h_angle')),
      cameraDistance: Number(this.getAttribute('cam_distance')),
      maxViewRange: Number(this.getAttribute('max_view_distance'))
    };
    
    console.log('[ThreeDModelViewer] 解析的配置:', config);
    return config;
  }

  applyContainerStyles(element) {
    element.style.cssText = `
      width: 100%;
      height: 100%;
      overflow: hidden;
      position: relative;
    `;
    console.log('[ThreeDModelViewer] 容器樣式已應用');
  }

  createLoadingOverlay() {
    console.log('[ThreeDModelViewer] 創建載入畫面');
    
    // 創建載入覆蓋層
    this.loadingOverlay = document.createElement('div');
    this.loadingOverlay.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      z-index: 1000;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: #e0e0e0;
      backdrop-filter: blur(10px);
    `;

    // 創建載入圖標（旋轉的圓環）
    const loadingIcon = document.createElement('div');
    loadingIcon.style.cssText = `
      width: 60px;
      height: 60px;
      border: 4px solid rgba(255, 255, 255, 0.1);
      border-top: 4px solid #64b5f6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin-bottom: 20px;
    `;

    // 添加旋轉動畫的 CSS
    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `;
    this._shadowRoot.appendChild(styleSheet);

    // 創建載入狀態文字
    this.loadingStatus = document.createElement('div');
    this.loadingStatus.style.cssText = `
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 20px;
      text-align: center;
    `;
    this.loadingStatus.textContent = '初始化 3D 檢視器...';

    // 創建進度條容器
    const progressContainer = document.createElement('div');
    progressContainer.style.cssText = `
      width: 300px;
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 10px;
      box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3);
    `;

    // 創建進度條
    this.progressBar = document.createElement('div');
    this.progressBar.style.cssText = `
      height: 100%;
      background: linear-gradient(90deg, #64b5f6, #42a5f5);
      width: 0%;
      transition: width 0.3s ease;
      border-radius: 4px;
      box-shadow: 0 1px 6px rgba(100, 181, 246, 0.4);
    `;

    // 創建進度文字
    this.progressText = document.createElement('div');
    this.progressText.style.cssText = `
      font-size: 14px;
      font-weight: 500;
      color: #b0b0b0;
    `;
    this.progressText.textContent = '0%';

    // 組裝載入畫面
    progressContainer.appendChild(this.progressBar);
    this.loadingOverlay.appendChild(loadingIcon);
    this.loadingOverlay.appendChild(this.loadingStatus);
    this.loadingOverlay.appendChild(progressContainer);
    this.loadingOverlay.appendChild(this.progressText);

    // 添加到容器
    this._shadowRoot.appendChild(this.loadingOverlay);
    
    console.log('[ThreeDModelViewer] 載入畫面創建完成');
  }

  updateLoadingProgress(percent, status = null) {
    if (this.progressBar) {
      this.progressBar.style.width = `${percent}%`;
    }
    
    if (this.progressText) {
      this.progressText.textContent = `${Math.round(percent)}%`;
    }
    
    if (status && this.loadingStatus) {
      this.loadingStatus.textContent = status;
    }
    
    console.log(`[ThreeDModelViewer] 載入進度更新: ${percent.toFixed(1)}%${status ? ' - ' + status : ''}`);
  }

  hideLoadingOverlay() {
    if (this.loadingOverlay) {
      // 添加淡出動畫
      this.loadingOverlay.style.transition = 'opacity 0.5s ease, visibility 0.5s ease';
      this.loadingOverlay.style.opacity = '0';
      
      setTimeout(() => {
        if (this.loadingOverlay && this.loadingOverlay.parentNode) {
          this.loadingOverlay.parentNode.removeChild(this.loadingOverlay);
          this.loadingOverlay = null;
          console.log('[ThreeDModelViewer] 載入畫面已移除');
        }
      }, 500);
    }
  }

  setup3DEnvironment(container, config) {
    console.log('[ThreeDModelViewer] 開始設置 3D 環境');
    
    // 更新載入進度 - 初始化階段
    this.updateLoadingProgress(10, '初始化 3D 環境...');
  
    // 打印 Three.js 版本資訊
    console.log('[ThreeDModelViewer] Three.js 版本:', THREE.REVISION);
    console.log('[ThreeDModelViewer] Three.js 完整資訊:', {
      revision: THREE.REVISION,
      version: `r${THREE.REVISION}`,
      buildDate: new Date().toISOString(),
      userAgent: navigator.userAgent
    });
    
    if (!container) {
      throw new Error('容器元素未找到');
    }

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;
    
    console.log(`[ThreeDModelViewer] 容器尺寸: ${width}x${height}`);

    // 初始化相機
    this.updateLoadingProgress(20, '初始化相機...');
    this.camera = new THREE.PerspectiveCamera(
      50, // 視角
      width / height, 
      1, // 近平面距離
      config.maxViewRange // 遠平面距離
    );
    console.log('[ThreeDModelViewer] 相機初始化完成');

    // 初始化渲染器
    this.updateLoadingProgress(30, '初始化渲染器...');
    this.renderEngine = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: false,
      preserveDrawingBuffer: true
    });
    this.renderEngine.setSize(width, height);
    container.appendChild(this.renderEngine.domElement);
    console.log('[ThreeDModelViewer] 渲染器初始化完成，Canvas 已添加');

    // 初始化場景
    this.updateLoadingProgress(40, '初始化場景...');
    this.sceneContainer = new THREE.Scene();
    console.log('[ThreeDModelViewer] 場景初始化完成');
    
    // 設置光照系統
    this.updateLoadingProgress(50, '設置光照系統...');
    this.setupLightingSystem();
    console.log('[ThreeDModelViewer] 光照系統設置完成');
    
    // 設置控制器
    this.updateLoadingProgress(60, '設置控制器...');
    this.setupCameraControls(config);
    console.log('[ThreeDModelViewer] 相機控制器設置完成');
    
    // 設置視窗調整
    this.updateLoadingProgress(65, '設置視窗調整...');
    this.setupResizeHandler(container);
    console.log('[ThreeDModelViewer] 視窗調整處理器設置完成');

    // 設置效能監視器引用
    this.setupPerformanceMonitor();

    // 立即開始基礎渲染
    this.startBasicRender();
    
    // 3D 環境設置完成
    this.updateLoadingProgress(70, '3D 環境設置完成，準備載入模型...');
  }

  setupPerformanceMonitor() {
    console.log('[ThreeDModelViewer] 正在設置效能監視器...');
    console.log('[ThreeDModelViewer] window.Stats 存在:', typeof window.Stats !== 'undefined');
    console.log('[ThreeDModelViewer] window.globalStats 存在:', typeof window.globalStats !== 'undefined');
    
    // 引用全域效能監視器
    if (window.globalStats) {
      this.stats = window.globalStats;
      console.log('[ThreeDModelViewer] ✅ 效能監視器已連接');
      console.log('[ThreeDModelViewer] Stats DOM 元素:', this.stats.dom);
    } else if (window.Stats) {
      // 如果全域 Stats 不存在，嘗試創建一個
      console.log('[ThreeDModelViewer] 全域 Stats 不存在，嘗試創建...');
      try {
        window.globalStats = new window.Stats();
        window.globalStats.showPanel(0);
        this.stats = window.globalStats;
        console.log('[ThreeDModelViewer] ✅ 效能監視器創建成功');
      } catch (error) {
        console.error('[ThreeDModelViewer] 效能監視器創建失敗:', error);
      }
    } else {
      console.warn('[ThreeDModelViewer] ❌ 效能監視器未找到 - Stats.js 可能未正確載入');
      
      // 延遲重試
      setTimeout(() => {
        console.log('[ThreeDModelViewer] 延遲重試連接效能監視器...');
        this.setupPerformanceMonitor();
      }, 500);
    }
  } 

  setupLightingSystem() {
    //光照設定
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.sceneContainer.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(-1, 1, 1);
    this.sceneContainer.add(dirLight);

    const bottomLight = new THREE.DirectionalLight(0xffffff, 1.0);
    bottomLight.position.set(-1, 1, -1);
    this.sceneContainer.add(bottomLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
    fillLight.position.set(1, -1, 1);
    this.sceneContainer.add(fillLight);


    console.log('[ThreeDModelViewer] 光照系統設置完成');
  }

  setupCameraControls(config) {
    //OrbitControls 設定
    this.cameraController = new THREE.OrbitControls(this.camera, this.renderEngine.domElement);
    this.cameraController.enableZoom = true;
    
    if (config.autoRotation) {
      this.cameraController.autoRotate = true;
      this.cameraController.autoRotateSpeed = 0.5;
      console.log('[ThreeDModelViewer] 自動旋轉已啟用');
    }
    
    console.log('[ThreeDModelViewer] OrbitControls 設置完成');
  }

  setupResizeHandler(container) {
    const resizeObserver = new ResizeObserver(() => {
      if (this.renderEngine && this.camera) {
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 600;
        
        this.renderEngine.setSize(width, height);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderFrame();
      }
    });
    resizeObserver.observe(container);
  }

  loadModel(config) {
    console.log('[ThreeDModelViewer] 開始載入模型:', config.modelPath);
    
    // 更新載入狀態
    this.updateLoadingProgress(75, '載入 3D 模型...');
    
    if (typeof THREE.STLLoader === 'undefined') {
      throw new Error('STLLoader 未載入');
    }

    const modelLoader = new THREE.STLLoader();
    
    modelLoader.load(
      config.modelPath, 
      (geometryData) => {
        console.log('[ThreeDModelViewer] 模型載入成功');
        this.updateLoadingProgress(90, '處理模型資料...');
        
        try {
          this.processGeometry(geometryData, config);
          console.log('[ThreeDModelViewer] 模型處理完成，開始渲染');
          
          // 完成載入
          this.updateLoadingProgress(100, '載入完成！');
          
          // 延遲一下讓使用者看到 100% 完成
          setTimeout(() => {
            this.hideLoadingOverlay();
            this.notifyRenderComplete();
          }, 800);
          
        } catch (error) {
          console.error('[ThreeDModelViewer] 模型處理失敗:', error);
          this.handleError(`模型處理失敗: ${error.message}`);
        }
      }, 
      (progress) => {
        const percent = progress.loaded && progress.total ? 
          (progress.loaded / progress.total * 100) : 0;
        
        // 將載入進度映射到 75-90% 之間
        const mappedPercent = 75 + (percent * 0.15);
        
        console.log(`[ThreeDModelViewer] 載入進度: ${percent.toFixed(1)}%`);
        this.updateLoadingProgress(mappedPercent, `載入模型中... ${percent.toFixed(1)}%`);
      },
      (error) => {
        console.error('[ThreeDModelViewer] 模型載入失敗:', error);
        this.handleError(`模型載入失敗: ${error}`);
      }
    );
  }

  processGeometry(geometry, config) {
    // 更新載入狀態
    this.updateLoadingProgress(92, '計算模型資料...');
    
    // 計算邊界框和幾何信息
    geometry.computeBoundingBox();
    geometry.computeVertexNormals(); // 確保法向量正確
    
    const boundingBox = geometry.boundingBox;
    const size = new THREE.Vector3();
    boundingBox.getSize(size);
    
    console.log('[ThreeDModelViewer] 模型尺寸:', size);
    console.log('[ThreeDModelViewer] 邊界框:', boundingBox);

    // 更新載入狀態
    this.updateLoadingProgress(94, '創建材質...');
    
    // 創建材質 - 使用高對比度顏色便於測試
    const material = this.createMaterial(config);
    
    // 創建網格
    this.modelMesh = new THREE.Mesh(geometry, material);
    
    // 更新載入狀態
    this.updateLoadingProgress(96, '處理幾何變換...');
    
    // 處理幾何變換
    this.processGeometryTransform(geometry, this.modelMesh, config);
    
    // 添加到場景
    this.sceneContainer.add(this.modelMesh);
    console.log('[ThreeDModelViewer] 網格已添加到場景');
    
    // 更新載入狀態
    this.updateLoadingProgress(98, '設置相機位置...');
    
    // 設置相機位置
    this.setupCameraPosition(geometry, config);
    
    // 開始渲染循環
    this.startRenderLoop();
  }

  createMaterial(config) {
    // 使用更明顯的材質設置
    const materialProps = {
      color: config.modelColor,
      opacity: config.materialOpacity,
      transparent: config.materialOpacity < 1.0,
      side: THREE.DoubleSide // 確保雙面渲染
    };

    let material;
    switch (config.renderMode) {
      case 'material':
        material = new THREE.MeshPhongMaterial({
          ...materialProps,
          shininess: config.surfaceShininess,
        });
        break;
      
      case 'flat':
        material = new THREE.MeshLambertMaterial({
          ...materialProps,
        });
        break;
      
      case 'wireframe':
        material = new THREE.MeshBasicMaterial({
          ...materialProps,
          wireframe: true,
          wireframeLinewidth: 2
        });
        break;
      
      default:
        material = new THREE.MeshPhongMaterial({
          ...materialProps,
          shininess: config.surfaceShininess,
        });
    }

    console.log('[ThreeDModelViewer] 材質創建完成:', material);
    return material;
  }

  processGeometryTransform(geometry, mesh, config) {
    //幾何變換處理
    const boundingBox = geometry.boundingBox;
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);
    
    // 使用 Matrix4 進行變換
    mesh.geometry.applyMatrix4(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
    
    // 避免奇點旋轉問題
    if (config.verticalAngle % 180 == 0) {
      mesh.rotation.z = config.horizontalAngle * (Math.PI / 180);
      console.log('[ThreeDModelViewer] 套用旋轉修正');
    }
    
    console.log('[ThreeDModelViewer] 幾何變換完成，中心點:', center);
  }

  setupCameraPosition(geometry, config) {
    const boundingBox = geometry.boundingBox;
    const size = new THREE.Vector3();
    boundingBox.getSize(size);

    console.log('[ThreeDModelViewer] 邊界框:', {
      min: boundingBox.min,
      max: boundingBox.max,
      size: size,
    });
    
    // 計算適當的相機距離
    const maxDimension = Math.max(size.x, size.y, size.z);
    console.log('[ThreeDModelViewer] 最大尺寸:', maxDimension);
    let distance = config.cameraDistance;
    
    if (distance === 0) {
      distance = maxDimension * 1.5;
      console.log('[ThreeDModelViewer] 自動計算相機距離:', distance);
    }

    // 相機位置設定
    const phi = config.verticalAngle * (Math.PI / 180);
    const theta = config.horizontalAngle * (Math.PI / 180);
    
    console.log('[ThreeDModelViewer] 相機位置計算設定:', {
      cam_v_angle: config.verticalAngle,
      cam_h_angle: config.horizontalAngle,
      phi: phi,
      theta: theta,
      x: distance * Math.sin(phi) * Math.cos(theta),
      y: distance * Math.sin(phi) * Math.sin(theta),
      z: distance * Math.cos(phi)
    });
    
    this.camera.position.x = distance * Math.sin(phi) * Math.cos(theta);
    this.camera.position.y = distance * Math.sin(phi) * Math.sin(theta);
    this.camera.position.z = distance * Math.cos(phi);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(new THREE.Vector3(0, 0, 0));
    
    console.log('[ThreeDModelViewer] 相機位置:', this.camera.position);
  }

  startBasicRender() {
    // 立即渲染一次確保畫面更新
    this.renderFrame();
    console.log('[ThreeDModelViewer] 基礎渲染已執行');
  }

  startRenderLoop() {
    console.log('[ThreeDModelViewer] 開始渲染循環');
    
    const animate = () => {
      if (!this.isActive) return;
      
      // 更新控制器
      if (this.cameraController) {
        this.cameraController.update();
      }
      
      this.renderFrame();
      requestAnimationFrame(animate);
    };
    
    animate();
  }

  renderFrame() {
    // 開始效能計時
    if (this.stats && typeof this.stats.begin === 'function') {
      this.stats.begin();
    }
    
    if (this.renderEngine && this.sceneContainer && this.camera) {
      this.renderEngine.render(this.sceneContainer, this.camera);
    }
    
    // 結束效能計時
    if (this.stats && typeof this.stats.end === 'function') {
      this.stats.end();
    }
  }

  notifyRenderComplete() {
    console.log('[ThreeDModelViewer] 渲染完成，發送通知');
    
    // 延遲發送事件確保渲染完成
    setTimeout(() => {
      const event = new CustomEvent('render-complete', {
        detail: { success: true }
      });
      this.dispatchEvent(event);
      
      // 移除 postMessage 以避免與 Streamlit 的訊息處理衝突
      // 如果需要通知父窗口，可以使用更具體的事件類型
      console.log('[ThreeDModelViewer] 渲染完成事件已發送');
    }, 100);
  }

  handleError(message) {
    console.error('[ThreeDModelViewer] 錯誤:', message);
    
    if (this._shadowRoot) {
      const errorDiv = document.createElement('div');
      errorDiv.style.cssText = `
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        height: 100%;
        color: #dc3545;
        font-family: Arial, sans-serif;
        font-size: 14px;
        text-align: center;
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        padding: 20px;
      `;
      errorDiv.innerHTML = `
        <div>
          <h3 style="margin: 0 0 10px 0;">載入錯誤</h3>
          <p style="margin: 0;">${message}</p>
        </div>
      `;
      
      this._shadowRoot.innerHTML = '';
      this._shadowRoot.appendChild(errorDiv);
    }

    // 移除 postMessage 以避免與 Streamlit 的訊息處理衝突
    // 錯誤已經在 console 中記錄，如有需要可以透過其他方式通知
    console.error('[ThreeDModelViewer] 錯誤已記錄，不發送 postMessage');
  }

  cleanup() {
    console.log('[ThreeDModelViewer] 開始清理資源');
    
    this.isActive = false;
    
    // 清理載入畫面
    if (this.loadingOverlay && this.loadingOverlay.parentNode) {
      this.loadingOverlay.parentNode.removeChild(this.loadingOverlay);
      this.loadingOverlay = null;
      this.progressBar = null;
      this.progressText = null;
      this.loadingStatus = null;
    }
    
    // 清理 Stats 引用
    this.stats = null;
    
    if (this.renderEngine) {
      this.renderEngine.dispose();
    }
    if (this.cameraController && this.cameraController.dispose) {
      this.cameraController.dispose();
    }
  }
}

// 註冊自定義元素
if (!customElements.get('three-d-model-viewer')) {
  customElements.define('three-d-model-viewer', ThreeDModelViewer);
  console.log('[ThreeDModelViewer] 自定義元素已註冊');
}