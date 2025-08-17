import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

// Initialize Firebase Admin SDK
let app;
if (!getApps().length) {
  // In production on GCP, this will automatically use the service account
  // In development, you might need to set GOOGLE_APPLICATION_CREDENTIALS
  app = initializeApp({
    projectId: process.env.GOOGLE_CLOUD_PROJECT || 'fmgen-net-production',
  });
} else {
  app = getApps()[0];
}

export const db = getFirestore(app);

// Configure Firestore to ignore undefined properties
db.settings({
  ignoreUndefinedProperties: true
});

// Test database connection
export async function testDbConnection(): Promise<boolean> {
  try {
    // Test Firestore by trying to get collection metadata
    await db.collection('health_check').limit(1).get();
    console.log('Firestore connection successful');
    return true;
  } catch (error) {
    console.error('Firestore connection failed:', error);
    return false;
  }
}

// Graceful shutdown
export async function closeDbConnection(): Promise<void> {
  try {
    // Firestore connections are automatically managed, no explicit close needed
    console.log('Firestore connection management handled automatically');
  } catch (error) {
    console.error('Error with Firestore connection:', error);
  }
}

// Health check
export async function checkDbHealth(): Promise<{
  connected: boolean;
  latency?: number;
  error?: string;
}> {
  try {
    const start = Date.now();
    // Test Firestore access
    await db.collection('health_check').limit(1).get();
    const latency = Date.now() - start;
    return { connected: true, latency };
  } catch (error) {
    return { 
      connected: false, 
      error: error instanceof Error ? error.message : 'Unknown error' 
    };
  }
}

// Initialize security settings
export async function initializeSecurity(): Promise<void> {
  try {
    const { EmailAuthorizationService } = await import('../services/email-authorization.service');
    const emailAuthService = new EmailAuthorizationService();
    await emailAuthService.initializeAuthorizedEmails();
    console.log('Security initialization completed');
  } catch (error) {
    console.error('Security initialization failed:', error);
    // Don't throw error to avoid breaking app startup
  }
}
