#!/bin/bash

echo "🚀 Starting JengaKsh Financial AI System"
echo "============================================"

# Set environment variables for development
export DATABASE_URL="postgres://user:password@localhost/jksh_financial"
export REDIS_URL="redis://localhost:6379"
export JWT_SECRET="jksh-financial-ai-secret-key-change-in-production"
export PORT=3000

# WhatsApp Cloud API Configuration (you need to set these)
export WHATSAPP_PHONE_NUMBER_ID="${WHATSAPP_PHONE_NUMBER_ID:-your_phone_number_id}"
export WHATSAPP_ACCESS_TOKEN="${WHATSAPP_ACCESS_TOKEN:-your_access_token}"
export WHATSAPP_VERIFY_TOKEN="${WHATSAPP_VERIFY_TOKEN:-jksh_verify_token_2024}"

# Financial Data API Keys (optional - will use mock data if not provided)
export ALPHA_VANTAGE_API_KEY="${ALPHA_VANTAGE_API_KEY:-demo}"
export FINNHUB_API_KEY="${FINNHUB_API_KEY:-demo}"
export POLYGON_API_KEY="${POLYGON_API_KEY:-demo}"

# Function to kill background processes on script exit
cleanup() {
    echo "🛑 Shutting down Financial AI system..."
    kill $BACKEND_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Check if we're in the correct directory
if [ ! -d "backend" ]; then
    echo "❌ Error: backend directory not found. Please run this script from the project root."
    exit 1
fi

# Build the backend in release mode for better performance
echo "🔨 Building Rust backend (release mode)..."
cd backend
if ! cargo build --release; then
    echo "❌ Failed to build backend"
    exit 1
fi

echo "📦 Starting Financial AI backend on port 3000..."
cargo run --release &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

echo ""
echo "✅ JengaKsh Financial AI System is running!"
echo ""
echo "🔗 API Endpoints:"
echo "   - Health Check: http://localhost:3000/api/health"
echo "   - Market Data:  http://localhost:3000/api/financial/market-data"
echo "   - AI Assistant: http://localhost:3000/api/financial/ai-recommendation"
echo ""
echo "📱 WhatsApp Integration:"
echo "   - Webhook URL: http://localhost:3000/webhook"
echo "   - Verify Token: $WHATSAPP_VERIFY_TOKEN"
echo ""
echo "💡 Test Commands:"
echo "   curl http://localhost:3000/api/health"
echo "   curl 'http://localhost:3000/api/financial/market-data?symbols=AAPL,GOOGL'"
echo ""
echo "📋 WhatsApp Commands to try:"
echo "   - Send 'help' to get the AI menu"
echo "   - Send 'price AAPL' to get Apple stock price"
echo "   - Send 'analyze TSLA' to get Tesla analysis"
echo "   - Send 'should I buy Microsoft?' for AI advice"
echo ""
echo "⚠️  Note: For WhatsApp functionality, make sure to:"
echo "   1. Set up your WhatsApp Business API account"
echo "   2. Configure webhook URL in Meta Developer Console"
echo "   3. Set the correct environment variables above"
echo ""
echo "Press Ctrl+C to stop the system"
echo ""

# Wait for background process
wait $BACKEND_PID