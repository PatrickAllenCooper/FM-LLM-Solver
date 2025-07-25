# Jupyter configuration for FM-LLM Solver development

c = get_config()

# Network settings
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Security settings
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True

# Notebook settings
c.ServerApp.notebook_dir = '/app'
c.ServerApp.allow_remote_access = True

# Extensions
c.LabApp.default_url = '/lab' 