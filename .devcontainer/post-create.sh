#!/usr/bin/env bash

# Ensure nvm is sourced in this script
export NVM_DIR="/usr/local/share/nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"                   # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" # This loads bash_completion for nvm

# Check if package.json exists, indicating a Node.js project
if [ -f package.json ]; then
  # Install Node.js LTS version if not already installed
  if ! nvm ls | grep -q "v16.0.0"; then
    echo "Installing Node.js LTS..."
    nvm install --lts
  fi
  # Install the latest npm version
  nvm install-latest-npm
  # Install project dependencies
  npm install
  # Run the build script
  npm run build
fi

# Install shfmt for shell script formatting if not already installed
echo "Installing shfmt..."
if ! command -v shfmt &>/dev/null; then
  curl -sS https://webi.sh/shfmt | sh -s -- --force &>/dev/null
fi

# Install Oh My Zsh if not already installed
if [ ! -d "$HOME/.oh-my-zsh" ]; then
  echo "Installing Oh My Zsh..."
  sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi

# Function to install Zsh plugins if missing
install_zsh_plugin() {
  local plugin_url=$1
  local plugin_name=$2
  if [ ! -d "$OMZ_CUSTOM/$plugin_name" ]; then
    echo "Installing plugin: $plugin_name"
    git clone "$plugin_url" "$OMZ_CUSTOM/$plugin_name"
  fi
}

# Define OMZ plugins directory
OMZ_CUSTOM="$HOME/.oh-my-zsh/custom/plugins"
mkdir -p "$OMZ_CUSTOM"

# Install Zsh plugins if missing
install_zsh_plugin "https://github.com/zsh-users/zsh-syntax-highlighting.git" "zsh-syntax-highlighting"
install_zsh_plugin "https://github.com/zsh-users/zsh-autosuggestions" "zsh-autosuggestions"

# Update ~/.zshrc to enable the plugins
echo "Enabling Zsh plugins in ~/.zshrc..."
sed -i -E "s/^(plugins=\()(git)(\))/\1\2 zsh-syntax-highlighting zsh-autosuggestions\3/" ~/.zshrc

# Prevent git log from using less
echo "Disabling 'less' for git log..."
grep -qxF "unset LESS" ~/.zshrc || echo "unset LESS" >>~/.zshrc
