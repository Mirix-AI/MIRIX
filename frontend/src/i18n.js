import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

const resources = {
  en: {
    translation: {
      tabs: {
        chat: 'Chat',
        screenshots: 'Screenshots',
        memory: 'Existing Memory',
        settings: 'Settings'
      },
      settings: {
        title: 'Settings',
        subtitle: 'Configure your MIRIX assistant',
        sections: {
          model: 'Model Configuration',
          preferences: 'Preferences',
          apiKeys: 'API Keys',
          about: 'About'
        },
        chatModel: 'Chat Agent Model',
        memoryModel: 'Memory Manager Model',
        persona: 'Persona',
        personaEdit: 'Edit',
        applyTemplate: 'Apply Template',
        editPersonaText: 'Edit Persona Text',
        buttons: {
          save: 'Save',
          cancel: 'Cancel'
        },
        language: 'Language',
        languageDescription: 'Select the display language',
        timezone: 'Timezone',
        apiKeyManagement: 'API Key Management',
        updateApiKeys: 'Update API Keys',
        about: {
          name: 'MIRIX Desktop',
          version: 'Version',
          docs: 'Documentation',
          reportIssue: 'Report Issue',
          description: 'AI Assistant powered by advanced language models'
        },
        add: 'Add',
        descriptions: {
          chatModel: 'Choose the AI model for chat responses',
          changingChatModel: 'Changing chat agent model...',
          memoryModel: 'Choose the AI model for memory management operations',
          changingMemoryModel: 'Changing memory manager model...',
          personaDisplay: "This shows the agent's current active persona. Click Edit to modify it.",
          personaEdit: 'Apply a template or customize the persona text to define how the assistant behaves.',
          loadingPersona: 'Loading persona...',
          templateSelector: 'Choose a template to load into the editor',
          loadingTemplate: 'Loading template...',
          personaPlaceholder: 'Enter your custom persona...',
          timezone: 'Your local timezone for timestamps',
          changingTimezone: 'Changing timezone...',
          apiKeyManagement: 'Configure and update your API keys for different AI models and services.',
          addModelTooltip: 'Add your own deployed model'
        },
        states: {
          saving: 'Saving...',
          updating: 'Updating...',
          applying: 'Applying...',
          changing: 'Changing...',
          checking: 'Checking...'
        }
      },
      chat: {
        model: 'Model',
        persona: 'Persona',
        screenshotTooltip: {
          enabled: 'Allow assistant to see your recent screenshots',
          disabled: 'Assistant cannot see your recent screenshots'
        },
        screenshotOn: 'ON',
        screenshotOff: 'OFF',
        stop: 'Stop',
        stopTitle: 'Stop generation',
        clear: 'Clear',
        clearTitle: 'Clear chat',
        welcome: {
          title: 'Welcome to MIRIX!',
          subtitle: 'Start a conversation with your AI assistant.',
          desktop: 'MIRIX is running in the desktop app environment.',
          web: 'Download the desktop app for an enhanced experience and more features!'
        },
        errorWithMessage: 'Error: {{message}}',
        clearFailed: 'Failed to clear conversation history',
        sender: {
          you: 'You',
          assistant: 'MIRIX',
          error: 'Error'
        },
        thinkingTitle: 'Thinking ...',
        steps_one: '({{count}} step)',
        steps_other: '({{count}} steps)',
        attachmentAlt: 'Attachment {{index}}'
      },
      messageInput: {
        removeFileTitle: 'Remove file',
        attachFilesTitle: 'Attach files',
        placeholder: 'Type your message... (Shift+Enter for new line)',
        sendTitle: 'Send message'
      },
      clearChat: {
        title: 'Clear Chat',
        choose: 'Choose how you want to clear the chat:',
        local: {
          title: '🗑️ Clear Current View',
          type: 'Local Only',
          desc: 'Clear the conversation display in this window. This only affects what you see here - your conversation history with the agent remains intact and memories are preserved.',
          button: 'Clear View Only'
        },
        permanent: {
          title: '⚠️ Clear All Conversation History',
          type: 'Permanent',
          desc: 'Permanently delete all conversation history between you and the chat agent. This cannot be undone. Your memories (episodic, semantic, etc.) will be preserved, but the chat history will be lost forever.',
          note: 'This action is permanent and cannot be undone!',
          button: 'Permanently Clear All',
          clearing: 'Clearing...'
        },
        cancel: 'Cancel'
      },
      screenshot: {
        title: 'Screen Monitor',
        controls: {
          openSystemPrefs: 'Open System Preferences',
          selectApps: 'Select Apps',
          permissionRequired: 'Permission Required',
          selectAppsFirst: 'Select Apps First',
          stopMonitor: 'Stop Monitor',
          startMonitor: 'Start Monitor',
          enhancedDetection: 'Accessibility Access'
        },
        status: {
          status: 'Status',
          permissions: 'Permissions',
          screenshotsSent: 'Screenshots sent',
          lastSent: 'Last sent',
          monitoring: 'monitoring',
          capturing: 'capturing',
          sending: 'sending',
          idle: 'idle',
          granted: 'Granted',
          denied: 'Denied',
          checking: 'Checking...'
        },
        monitoring: {
          multipleApps: 'Monitoring {{count}} apps',
          singleApp: 'Monitoring {{appName}}',
          noAppsVisible: 'No apps visible',
          statusInfo: 'Status: {{status}}',
          appsVisible: '{{visible}}/{{total}} apps visible ({{sent}} sent)',
          fullScreen: 'Full Screen',
          zoomScreenSharing: 'Full Screen - Zoom Screen Sharing',
          googleMeetScreenSharing: 'Full Screen - Google Meet Screen Sharing'
        },
        errors: {
          desktopOnly: 'Screenshot functionality is only available in the desktop app',
          permissionDenied: 'Screen recording permission not granted. Please grant screen recording permissions in System Preferences > Security & Privacy > Screen Recording and restart the application.',
          permissionCheckFailed: 'Permission check failed: {{error}}',
          systemPrefsOnly: 'System Preferences functionality is only available in the desktop app',
          systemPrefsFailed: 'Failed to open System Preferences',
          systemPrefsError: 'Failed to open System Preferences: {{error}}',
          screenshotProcessing: 'Error processing screenshot: {{error}}',
          screenshotFailed: 'Failed to send screenshot: {{error}}',
          screenshotsFailed: 'Failed to send screenshots: {{error}}',
          desktopRequired: 'Screenshot functionality requires desktop app',
          enhancedPermissionsNotAvailable: 'Accessibility permissions functionality not available',
          enhancedPermissionsDenied: 'Enhanced screen sharing detection requires Accessibility permissions. Please enable MIRIX in System Preferences > Privacy & Security > Accessibility.',
          enhancedPermissionsFailed: 'Failed to request accessibility permissions',
          enhancedPermissionsError: 'Error requesting accessibility permissions: {{error}}'
        },
        permissions: {
          warningTitle: 'Screen recording permission is required to use the screen monitor feature.',
          warningAction: 'Click "⚙️ Open System Preferences" to grant permission directly!',
          helpTitle: 'How to grant permission:',
          helpStep1: '1. Click "⚙️ Open System Preferences" button above',
          helpStep2: '2. Find "MIRIX" in the list and check the box next to it',
          helpStep3: '3. No restart required - permissions take effect immediately'
        }
      },
      appSelector: {
        title: 'Select Apps to Monitor',
        loading: 'Scanning available apps and windows...',
        filters: {
          all: 'All',
          windows: 'Windows',
          screens: 'Screens'
        },
        types: {
          window: 'window',
          screen: 'screen'
        },
        status: {
          hidden: 'Hidden',
          hiddenTooltip: 'This window is minimized or on another desktop'
        },
        footer: {
          sourcesSelected_one: '{{count}} source selected',
          sourcesSelected_other: '{{count}} sources selected',
          cancel: 'Cancel',
          startMonitoring: 'Start Monitoring'
        },
        errors: {
          desktopOnly: 'App selection is only available in the desktop app',
          failedToLoad: 'Failed to get capture sources',
          loadError: 'Failed to load sources: {{error}}'
        }
      },
      localModel: {
        title: 'Add Local Model',
        form: {
          modelName: 'Model Name',
          modelNamePlaceholder: 'e.g. qwen3-32b',
          modelNameDescription: 'The name identifier for your deployed model',
          modelEndpoint: 'Model Endpoint',
          modelEndpointPlaceholder: 'e.g. http://localhost:47283/v1',
          modelEndpointDescription: 'The API endpoint URL for your deployed model',
          apiKey: 'API Key',
          apiKeyDescription: 'Authentication key for your model endpoint',
          temperature: 'Temperature',
          temperatureDescription: 'Controls randomness in responses (0.0 = deterministic, 1.0 = creative)',
          maxTokens: 'Max Tokens',
          maxTokensDescription: 'Maximum number of tokens to generate in each response',
          maximumLength: 'Maximum Length',
          maximumLengthDescription: 'Maximum context length supported by the model',
          required: '*',
          cancel: 'Cancel',
          addModel: 'Add Model',
          adding: 'Adding...'
        },
        errors: {
          modelNameRequired: 'Model name is required',
          endpointRequired: 'Model endpoint is required',
          apiKeyRequired: 'API key is required'
        }
      },
      memory: {
        types: {
          episodic: 'Episodic',
          semantic: 'Semantic',
          procedural: 'Procedural',
          resource: 'Resource',
          core: 'Core',
          credentials: 'Credentials'
        },
        tabs: {
          'past-events': 'Past Events',
          'semantic': 'Semantic',
          'procedural': 'Procedural',
          'docs-files': 'Docs & Files',
          'core-understanding': 'Core Understanding',
          'credentials': 'Credentials'
        },
        actions: {
          uploadExport: 'Upload & Export',
          reflexion: 'Reflexion',
          processing: 'Processing...',
          refresh: 'Refresh',
          retry: 'Retry',
          edit: 'Edit',
          save: 'Save',
          cancel: 'Cancel',
          saving: 'Saving...',
          clearSearch: 'Clear search',
          expandDetails: 'Expand details',
          collapseDetails: 'Collapse details',
          hideDetails: 'Hide Details',
          showDetails: 'Show Details'
        },
        search: {
          placeholder: 'Search {{type}}...',
          noResults: 'No {{type}} found matching "{{query}}".',
          tryDifferent: 'Try a different search term or clear the search to see all memories.',
          clearToSeeAll: 'Clear search to see all memories'
        },
        view: {
          listView: 'List',
          treeView: 'Tree'
        },
        states: {
          loading: 'Loading memory data...',
          error: 'Error loading memory: {{error}}',
          empty: 'No {{type}} found.',
          loadingTree: 'Loading {{type}} memory tree...',
          treeError: 'Error: {{error}}'
        },
        details: {
          updated: 'Updated: {{date}}',
          lastAccessed: 'Last accessed: {{date}}',
          lastPracticed: 'Last Practiced: {{date}}',
          size: 'Size: {{size}}',
          characterCount: '{{current}}/{{max}} characters',
          editing: '(editing)',
          proficiency: 'Proficiency: {{value}}',
          difficulty: 'Difficulty: {{value}}',
          successRate: 'Success Rate: {{value}}',
          timeToComplete: 'Time to Complete: {{value}}',
          prerequisites: 'Prerequisites: {{list}}',
          stepByStepGuide: 'Step-by-Step Guide:',
          noStepsAvailable: 'No steps available',
          unknownTime: 'Unknown time',
          unknownType: 'Unknown',
          enterCoreUnderstanding: 'Enter core understanding...',
          credentialType: 'Credential',
          credentialMasked: 'Content masked for security',
          source: 'Source: {{source}}',
          sensitivity: '{{level}} Sensitivity'
        },
        reflexion: {
          title: 'Reorganize memory with Reflexion Agent',
          success: 'Reflexion completed successfully',
          error: 'Reflexion failed'
        },
        tooltips: {
          uploadExport: 'Upload & Export Memory Data',
          reflexion: 'Reorganize memory with Reflexion Agent',
          listView: 'List view',
          treeView: 'Tree view'
        }
      },
      uploadExport: {
        title: 'Memory Upload & Export',
        sections: {
          upload: 'Upload Memory Data',
          export: 'Export Memory Data'
        },
        memoryTypes: {
          episodic: 'Episodic',
          semantic: 'Semantic', 
          procedural: 'Procedural',
          resource: 'Resource'
        },
        memoryTypeDescriptions: {
          episodic: 'Personal experiences and events',
          semantic: 'Facts and general knowledge',
          procedural: 'Skills and procedures',
          resource: 'Files and documents'
        },
        form: {
          selectTypes: 'Select Memory Types to Export:',
          exportPath: 'Export File Path:',
          browse: 'Browse',
          pathPlaceholder: 'Enter file path for export...',
          upload: 'Upload',
          export: 'Export',
          exporting: 'Exporting...',
          close: 'Close'
        },
        descriptions: {
          modalDescription: 'Manage your memory data - upload new data or export existing memories',
          uploadSection: 'Import memory data from external sources',
          exportSection: 'Export selected memory types to Excel with separate sheets',
          saveDialogTitle: 'Save Memory Export',
          defaultFileName: 'memories_export.xlsx'
        },
        alerts: {
          pathRequired: 'Please enter or browse for a file path for export',
          selectTypes: 'Please select at least one memory type to export',
          uploadNotImplemented: 'Upload functionality is not implemented yet (mock feature)',
          browserUnavailable: 'File browser not available. Please enter the path manually.',
          browserFailed: 'Failed to open file browser. Please enter the path manually.',
          exportFailed: 'Export failed'
        },
        status: {
          success: 'Export completed successfully!',
          failed: 'Export failed',
          exported: 'Total exported: {{total}} items',
          breakdown: 'Breakdown: {{breakdown}}'
        },
        errors: {
          atLeastOneSheetVisible: 'At least one sheet must be visible',
          noData: 'No data to export',
          permissionDenied: 'Permission denied when writing the file',
          unknown: 'Export failed'
        }
      }
    }
  },
  zh: {
    translation: {
      tabs: {
        chat: '聊天',
        screenshots: '截图',
        memory: '记忆库',
        settings: '设置'
      },
      settings: {
        title: '设置',
        subtitle: '配置你的 MIRIX 助手',
        sections: {
          model: '模型配置',
          preferences: '偏好设置',
          apiKeys: 'API 密钥',
          about: '关于'
        },
        chatModel: '聊天模型',
        memoryModel: '记忆管理模型',
        persona: '人设',
        personaEdit: '编辑',
        applyTemplate: '应用模板',
        editPersonaText: '编辑人设文本',
        buttons: {
          save: '保存',
          cancel: '取消'
        },
        language: '语言',
        languageDescription: '选择界面语言',
        timezone: '时区',
        apiKeyManagement: 'API 密钥管理',
        updateApiKeys: '更新 API 密钥',
        about: {
          name: 'MIRIX 桌面端',
          version: '版本',
          docs: '文档',
          reportIssue: '反馈问题',
          description: '由先进语言模型驱动的AI助手'
        },
        add: '添加',
        descriptions: {
          chatModel: '选择用于聊天回复的AI模型',
          changingChatModel: '正在更改聊天代理模型...',
          memoryModel: '选择用于记忆管理操作的AI模型',
          changingMemoryModel: '正在更改记忆管理模型...',
          personaDisplay: '这里显示助手当前的活跃人设。点击编辑来修改。',
          personaEdit: '应用模板或自定义人设文本来定义助手的行为方式。',
          loadingPersona: '正在加载人设...',
          templateSelector: '选择一个模板加载到编辑器中',
          loadingTemplate: '正在加载模板...',
          personaPlaceholder: '输入你的自定义人设...',
          timezone: '用于时间戳的本地时区',
          changingTimezone: '正在更改时区...',
          apiKeyManagement: '为不同的AI模型和服务配置和更新你的API密钥。',
          addModelTooltip: '添加你自己部署的模型'
        },
        states: {
          saving: '保存中...',
          updating: '更新中...',
          applying: '应用中...',
          changing: '更改中...',
          checking: '检查中...'
        }
      },
      chat: {
        model: '模型',
        persona: '人设',
        screenshotTooltip: {
          enabled: '允许助手查看你最近的截图',
          disabled: '助手将无法查看你最近的截图'
        },
        screenshotOn: '开',
        screenshotOff: '关',
        stop: '停止',
        stopTitle: '停止生成',
        clear: '清空',
        clearTitle: '清空对话',
        welcome: {
          title: '欢迎使用 MIRIX！',
          subtitle: '开始与 AI 助手对话。',
          desktop: 'MIRIX 正在桌面端环境运行。',
          web: '下载桌面版以获得更好的体验和更多功能！'
        },
        errorWithMessage: '错误：{{message}}',
        clearFailed: '清空对话历史失败',
        sender: {
          you: '你',
          assistant: 'MIRIX',
          error: '错误'
        },
        thinkingTitle: '思考中 ...',
        steps_one: '（{{count}} 步）',
        steps_other: '（{{count}} 步）',
        attachmentAlt: '附件 {{index}}'
      },
      messageInput: {
        removeFileTitle: '移除文件',
        attachFilesTitle: '添加文件',
        placeholder: '输入消息...（Shift+Enter 换行）',
        sendTitle: '发送消息'
      },
      clearChat: {
        title: '清空对话',
        choose: '选择清空对话的方式：',
        local: {
          title: '🗑️ 清空当前视图',
          type: '仅本地',
          desc: '清空当前窗口中的对话显示。该操作仅影响你在此处看到的内容——与你和助手之间的对话历史仍会保留，记忆不会被删除。',
          button: '仅清空视图'
        },
        permanent: {
          title: '⚠️ 清空全部对话历史',
          type: '永久',
          desc: '永久删除你与聊天助手之间的所有对话历史。该操作不可撤销。你的记忆（情景记忆、语义记忆等）将被保留，但对话历史将被永久清除。',
          note: '此操作不可撤销！',
          button: '永久清空全部',
          clearing: '清理中...'
        },
        cancel: '取消'
      },
      screenshot: {
        title: '屏幕监控',
        controls: {
          openSystemPrefs: '打开系统偏好设置',
          selectApps: '选择应用',
          permissionRequired: '需要权限',
          selectAppsFirst: '请先选择应用',
          stopMonitor: '停止监控',
          startMonitor: '开始监控',
          enhancedDetection: '辅助功能权限'
        },
        status: {
          status: '状态',
          permissions: '权限',
          screenshotsSent: '已发送截图',
          lastSent: '最后发送',
          monitoring: '监控中',
          capturing: '截图中',
          sending: '发送中',
          idle: '空闲',
          granted: '已授权',
          denied: '被拒绝',
          checking: '检查中...'
        },
        monitoring: {
          multipleApps: '监控 {{count}} 个应用',
          singleApp: '监控 {{appName}}',
          noAppsVisible: '没有可见应用',
          statusInfo: '状态：{{status}}',
          appsVisible: '{{visible}}/{{total}} 个应用可见（已发送 {{sent}} 张）',
          fullScreen: '全屏',
          zoomScreenSharing: '全屏 - Zoom 屏幕共享',
          googleMeetScreenSharing: '全屏 - Google Meet 屏幕共享'
        },
        errors: {
          desktopOnly: '截图功能仅在桌面端应用中可用',
          permissionDenied: '未授予屏幕录制权限。请在系统偏好设置 > 安全性与隐私 > 屏幕录制中授予屏幕录制权限并重启应用程序。',
          permissionCheckFailed: '权限检查失败：{{error}}',
          systemPrefsOnly: '系统偏好设置功能仅在桌面端应用中可用',
          systemPrefsFailed: '打开系统偏好设置失败',
          systemPrefsError: '打开系统偏好设置失败：{{error}}',
          screenshotProcessing: '处理截图时出错：{{error}}',
          screenshotFailed: '发送截图失败：{{error}}',
          screenshotsFailed: '发送截图失败：{{error}}',
          desktopRequired: '截图功能需要桌面端应用',
          enhancedPermissionsNotAvailable: '增强权限功能不可用',
          enhancedPermissionsDenied: '增强屏幕共享检测需要额外权限。请在系统偏好设置中授予访问权限。',
          enhancedPermissionsFailed: '请求增强权限失败',
          enhancedPermissionsError: '请求增强权限时出错：{{error}}'
        },
        permissions: {
          warningTitle: '需要屏幕录制权限才能使用屏幕监控功能。',
          warningAction: '点击"⚙️ 打开系统偏好设置"按钮直接授权！',
          helpTitle: '如何授予权限：',
          helpStep1: '1. 点击上方的"⚙️ 打开系统偏好设置"按钮',
          helpStep2: '2. 在列表中找到"MIRIX"并勾选旁边的复选框',
          helpStep3: '3. 无需重启 - 权限立即生效'
        }
      },
      appSelector: {
        title: '选择要监控的应用',
        loading: '正在扫描可用的应用和窗口...',
        filters: {
          all: '全部',
          windows: '窗口',
          screens: '屏幕'
        },
        types: {
          window: '窗口',
          screen: '屏幕'
        },
        status: {
          hidden: '已隐藏',
          hiddenTooltip: '此窗口已最小化或在其他桌面上'
        },
        footer: {
          sourcesSelected_one: '已选择 {{count}} 个源',
          sourcesSelected_other: '已选择 {{count}} 个源',
          cancel: '取消',
          startMonitoring: '开始监控'
        },
        errors: {
          desktopOnly: '应用选择功能仅在桌面端应用中可用',
          failedToLoad: '获取捕获源失败',
          loadError: '加载源失败：{{error}}'
        }
      },
      localModel: {
        title: '添加本地模型',
        form: {
          modelName: '模型名称',
          modelNamePlaceholder: '例如：qwen3-32b',
          modelNameDescription: '你部署的模型的名称标识符',
          modelEndpoint: '模型端点',
          modelEndpointPlaceholder: '例如：http://localhost:47283/v1',
          modelEndpointDescription: '你部署的模型的API端点URL',
          apiKey: 'API 密钥',
          apiKeyDescription: '模型端点的认证密钥',
          temperature: '温度',
          temperatureDescription: '控制响应的随机性（0.0 = 确定性，1.0 = 创意性）',
          maxTokens: '最大Token数',
          maxTokensDescription: '每次响应中生成的最大token数量',
          maximumLength: '最大长度',
          maximumLengthDescription: '模型支持的最大上下文长度',
          required: '*',
          cancel: '取消',
          addModel: '添加模型',
          adding: '添加中...'
        },
        errors: {
          modelNameRequired: '模型名称是必填项',
          endpointRequired: '模型端点是必填项',
          apiKeyRequired: 'API密钥是必填项'
        }
      },
      memory: {
        types: {
          episodic: '情景记忆',
          semantic: '语义记忆',
          procedural: '程序记忆',
          resource: '资源记忆',
          core: '核心记忆',
          credentials: '凭据记忆'
        },
        tabs: {
          'past-events': '过往事件',
          'semantic': '语义记忆',
          'procedural': '程序记忆',
          'docs-files': '文档和文件',
          'core-understanding': '核心理解',
          'credentials': '凭据'
        },
        actions: {
          uploadExport: '上传和导出',
          reflexion: '反思',
          processing: '处理中...',
          refresh: '刷新',
          retry: '重试',
          edit: '编辑',
          save: '保存',
          cancel: '取消',
          saving: '保存中...',
          clearSearch: '清除搜索',
          expandDetails: '展开详情',
          collapseDetails: '折叠详情',
          hideDetails: '隐藏详情',
          showDetails: '显示详情'
        },
        search: {
          placeholder: '搜索{{type}}...',
          noResults: '未找到匹配"{{query}}"的{{type}}。',
          tryDifferent: '尝试不同的搜索词或清除搜索以查看所有记忆。',
          clearToSeeAll: '清除搜索以查看所有记忆'
        },
        view: {
          listView: '列表',
          treeView: '树形'
        },
        states: {
          loading: '正在加载记忆数据...',
          error: '加载记忆时出错：{{error}}',
          empty: '未找到{{type}}。',
          loadingTree: '正在加载{{type}}记忆树...',
          treeError: '错误：{{error}}'
        },
        details: {
          updated: '更新时间：{{date}}',
          lastAccessed: '最后访问：{{date}}',
          lastPracticed: '最后练习：{{date}}',
          size: '大小：{{size}}',
          characterCount: '{{current}}/{{max}} 字符',
          editing: '（编辑中）',
          proficiency: '熟练度：{{value}}',
          difficulty: '难度：{{value}}',
          successRate: '成功率：{{value}}',
          timeToComplete: '完成时间：{{value}}',
          prerequisites: '前置条件：{{list}}',
          stepByStepGuide: '分步指南：',
          noStepsAvailable: '无可用步骤',
          unknownTime: '未知时间',
          unknownType: '未知',
          enterCoreUnderstanding: '输入核心理解...',
          credentialType: '凭据',
          credentialMasked: '出于安全考虑，内容已隐藏',
          source: '来源：{{source}}',
          sensitivity: '{{level}}敏感度'
        },
        reflexion: {
          title: '使用反思代理重组记忆',
          success: '反思成功完成',
          error: '反思失败'
        },
        tooltips: {
          uploadExport: '上传和导出记忆数据',
          reflexion: '使用反思代理重组记忆',
          listView: '列表视图',
          treeView: '树形视图'
        }
      },
      uploadExport: {
        title: '记忆上传和导出',
        sections: {
          upload: '上传记忆数据',
          export: '导出记忆数据'
        },
        memoryTypes: {
          episodic: '情景记忆',
          semantic: '语义记忆',
          procedural: '程序记忆',
          resource: '资源记忆'
        },
        memoryTypeDescriptions: {
          episodic: '个人经历和事件',
          semantic: '事实和常识',
          procedural: '技能和程序',
          resource: '文件和文档'
        },
        form: {
          selectTypes: '选择要导出的记忆类型：',
          exportPath: '导出文件路径：',
          browse: '浏览',
          pathPlaceholder: '输入导出文件路径...',
          upload: '上传',
          export: '导出',
          exporting: '导出中...',
          close: '关闭'
        },
        descriptions: {
          modalDescription: '管理您的记忆数据 - 上传新数据或导出现有记忆',
          uploadSection: '从外部来源导入记忆数据',
          exportSection: '将选定的记忆类型导出到Excel表格的不同工作表中',
          saveDialogTitle: '保存记忆导出',
          defaultFileName: 'memories_export.xlsx'
        },
        alerts: {
          pathRequired: '请输入或浏览选择导出文件路径',
          selectTypes: '请至少选择一种记忆类型进行导出',
          uploadNotImplemented: '上传功能尚未实现（模拟功能）',
          browserUnavailable: '文件浏览器不可用。请手动输入路径。',
          browserFailed: '无法打开文件浏览器。请手动输入路径。',
          exportFailed: '导出失败'
        },
        status: {
          success: '导出成功完成！',
          failed: '导出失败',
          exported: '总计导出：{{total}} 项',
          breakdown: '详细：{{breakdown}}'
        },
        errors: {
          atLeastOneSheetVisible: '至少需要一个工作表可见',
          noData: '没有可导出的数据',
          permissionDenied: '无权限写入文件',
          unknown: '导出失败'
        }
      }
    }
  }
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage']
    }
  });

export default i18n; 