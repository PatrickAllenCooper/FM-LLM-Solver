// global-setup.js
const { chromium } = require('@playwright/test');

async function globalSetup() {
  console.log('🚀 Setting up performance test environment...');
  
  // Create a browser instance for setup
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    // Check if the application is running
    const baseURL = process.env.BASE_URL || 'http://localhost:5000';
    console.log(`📡 Checking application at ${baseURL}`);
    
    await page.goto(baseURL, { timeout: 30000 });
    
    // Wait for the application to be ready
    await page.waitForSelector('body', { timeout: 10000 });
    
    console.log('✅ Application is ready for testing');
    
    // Create test admin user if needed
    try {
      await page.goto(`${baseURL}/auth/register`);
      
      const adminUser = {
        username: 'perf_test_admin',
        email: 'perftest@example.com',
        password: 'TestPassword123!'
      };
      
      await page.fill('[name="username"]', adminUser.username);
      await page.fill('[name="email"]', adminUser.email);
      await page.fill('[name="password"]', adminUser.password);
      await page.fill('[name="confirm_password"]', adminUser.password);
      
      await page.click('[type="submit"]');
      
      // Check if registration was successful
      try {
        await page.waitForURL('**/auth/profile', { timeout: 5000 });
        console.log('✅ Test admin user created');
      } catch (e) {
        console.log('ℹ️  Test admin user already exists or login required');
      }
    } catch (error) {
      console.log('ℹ️  Could not create test admin user (may already exist)');
    }
    
    // Pre-warm the application
    console.log('🔥 Pre-warming application...');
    
    // Make several requests to warm up caches
    const warmupRoutes = ['/', '/auth/login', '/api/health'];
    
    for (const route of warmupRoutes) {
      try {
        await page.goto(`${baseURL}${route}`);
        await page.waitForLoadState('networkidle');
      } catch (error) {
        console.log(`⚠️  Could not warm up route ${route}: ${error.message}`);
      }
    }
    
    console.log('✅ Application pre-warming complete');
    
    // Set up test data if needed
    await setupTestData(page, baseURL);
    
  } catch (error) {
    console.error('❌ Global setup failed:', error.message);
    throw error;
  } finally {
    await browser.close();
  }
  
  console.log('🎯 Performance test environment setup complete');
}

async function setupTestData(page, baseURL) {
  console.log('📊 Setting up test data...');
  
  try {
    // Login as admin user
    await page.goto(`${baseURL}/auth/login`);
    await page.fill('[name="email"]', 'perftest@example.com');
    await page.fill('[name="password"]', 'TestPassword123!');
    await page.click('[type="submit"]');
    
    // Generate a few sample certificates for testing
    await page.goto(baseURL);
    
    const testSystems = [
      'dx/dt = -x + u',
      'dx/dt = -2*x + u, u ∈ [-1, 1]',
      'dx/dt = A*x + B*u, A = [-1, 0; 0, -2], B = [1; 1]'
    ];
    
    for (let i = 0; i < testSystems.length; i++) {
      try {
        await page.fill('[name="system_description"]', testSystems[i]);
        await page.fill('[name="domain_bounds"]', 'x ∈ [-5, 5]');
        
        await page.click('[data-testid="generate-certificate"]');
        
        // Wait a bit for generation (but don't wait for completion to save time)
        await page.waitForTimeout(2000);
        
        // Navigate back to home for next generation
        await page.goto(baseURL);
      } catch (error) {
        console.log(`⚠️  Could not generate test certificate ${i + 1}`);
      }
    }
    
    console.log('✅ Test data setup complete');
    
  } catch (error) {
    console.log('⚠️  Could not set up test data:', error.message);
  }
}

module.exports = globalSetup; 