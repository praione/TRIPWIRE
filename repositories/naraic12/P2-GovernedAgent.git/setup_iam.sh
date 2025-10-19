#!/bin/bash
# IAM Configuration Script for Project Resilience Secret Manager Access
# This script sets up the correct IAM roles for Constitutional Tripwire components

# ================================================
# CONFIGURATION VARIABLES
# ================================================

PROJECT_ID="project-resilience-ai-one"
SECRET_NAME="constitutional-tripwire-polynomial"

# Service accounts (create these if they don't exist)
VETO_MANAGER_SA="veto-manager@${PROJECT_ID}.iam.gserviceaccount.com"
SUBLIMINAL_PROOF_SA="subliminal-proof@${PROJECT_ID}.iam.gserviceaccount.com"
ARBITER_SA="arbiter-agent@${PROJECT_ID}.iam.gserviceaccount.com"
DISPATCHER_SA="dispatcher@${PROJECT_ID}.iam.gserviceaccount.com"

# Your user account (for development/testing)
USER_EMAIL="support@ciarandoyle.com"  # Replace with your actual email

# ================================================
# STEP 1: CREATE SERVICE ACCOUNTS
# ================================================

echo "Creating service accounts..."

# Create service account for veto_manager (needs write access)
gcloud iam service-accounts create veto-manager \
    --display-name="Veto Manager Service Account" \
    --description="Manages constitutional tripwire veto keys" \
    --project="${PROJECT_ID}"

# Create service account for subliminal_proof (needs read access)
gcloud iam service-accounts create subliminal-proof \
    --display-name="Subliminal Proof Service Account" \
    --description="Reads polynomial coefficients for share generation" \
    --project="${PROJECT_ID}"

# Create service account for arbiter (needs read access)
gcloud iam service-accounts create arbiter-agent \
    --display-name="Arbiter Agent Service Account" \
    --description="Monitors and reconstructs veto keys" \
    --project="${PROJECT_ID}"

# Create service account for dispatcher (needs read access)
gcloud iam service-accounts create dispatcher \
    --display-name="Dispatcher Service Account" \
    --description="Main orchestrator with tripwire integration" \
    --project="${PROJECT_ID}"

echo "Service accounts created."

# ================================================
# STEP 2: CREATE THE SECRET (if it doesn't exist)
# ================================================

echo "Creating secret if it doesn't exist..."

gcloud secrets create ${SECRET_NAME} \
    --data-file=- \
    --replication-policy="automatic" \
    --project="${PROJECT_ID}" <<EOF
{
    "polynomial_coefficients": [1234567890, 9876543210, 5555555555],
    "threshold": 3,
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "1.0"
}
EOF

echo "Secret created or already exists."

# ================================================
# STEP 3: ASSIGN IAM ROLES
# ================================================

echo "Assigning IAM roles..."

# VETO MANAGER - Needs full access (create, update, delete secrets)
gcloud secrets add-iam-policy-binding ${SECRET_NAME} \
    --member="serviceAccount:${VETO_MANAGER_SA}" \
    --role="roles/secretmanager.admin" \
    --project="${PROJECT_ID}"

# SUBLIMINAL PROOF - Needs read access only
gcloud secrets add-iam-policy-binding ${SECRET_NAME} \
    --member="serviceAccount:${SUBLIMINAL_PROOF_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="${PROJECT_ID}"

# ARBITER - Needs read access only
gcloud secrets add-iam-policy-binding ${SECRET_NAME} \
    --member="serviceAccount:${ARBITER_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="${PROJECT_ID}"

# DISPATCHER - Needs read access only
gcloud secrets add-iam-policy-binding ${SECRET_NAME} \
    --member="serviceAccount:${DISPATCHER_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="${PROJECT_ID}"

# YOUR USER ACCOUNT - Full access for development
gcloud secrets add-iam-policy-binding ${SECRET_NAME} \
    --member="user:${USER_EMAIL}" \
    --role="roles/secretmanager.admin" \
    --project="${PROJECT_ID}"

echo "IAM roles assigned."

# ================================================
# STEP 4: CREATE KEY FILES FOR LOCAL DEVELOPMENT
# ================================================

echo "Creating service account key files for local development..."

# Create keys directory
mkdir -p keys

# Generate key for veto_manager
gcloud iam service-accounts keys create keys/veto-manager-key.json \
    --iam-account=${VETO_MANAGER_SA} \
    --project="${PROJECT_ID}"

# Generate key for subliminal_proof
gcloud iam service-accounts keys create keys/subliminal-proof-key.json \
    --iam-account=${SUBLIMINAL_PROOF_SA} \
    --project="${PROJECT_ID}"

# Generate key for arbiter
gcloud iam service-accounts keys create keys/arbiter-key.json \
    --iam-account=${ARBITER_SA} \
    --project="${PROJECT_ID}"

# Generate key for dispatcher
gcloud iam service-accounts keys create keys/dispatcher-key.json \
    --iam-account=${DISPATCHER_SA} \
    --project="${PROJECT_ID}"

echo "Service account keys created in ./keys directory."

# ================================================
# STEP 5: SET UP APPLICATION DEFAULT CREDENTIALS
# ================================================

echo ""
echo "To use these service accounts in your application:"
echo ""
echo "For veto_manager.py (write access):"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=keys/veto-manager-key.json"
echo ""
echo "For subliminal_proof.py (read access):"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=keys/subliminal-proof-key.json"
echo ""
echo "For arbiter_agent.py (read access):"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=keys/arbiter-key.json"
echo ""
echo "For dispatcher.py (read access):"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=keys/dispatcher-key.json"
echo ""
echo "Or for development with your user account:"
echo "  gcloud auth application-default login"

# ================================================
# STEP 6: VERIFY CONFIGURATION
# ================================================

echo ""
echo "Verifying configuration..."

# Test read access
echo "Testing read access..."
gcloud secrets versions access latest \
    --secret=${SECRET_NAME} \
    --project="${PROJECT_ID}"

if [ $? -eq 0 ]; then
    echo "✓ Read access verified"
else
    echo "✗ Read access failed"
fi

# List all IAM bindings for the secret
echo ""
echo "Current IAM bindings for secret '${SECRET_NAME}':"
gcloud secrets get-iam-policy ${SECRET_NAME} \
    --project="${PROJECT_ID}" \
    --format="table(bindings.role,bindings.members)"

echo ""
echo "IAM configuration complete!"