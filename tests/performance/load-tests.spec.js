// @ts-check
const { test, expect } = require('@playwright/test');
const { performance } = require('perf_hooks');

/**
 * Load Testing Suite for FM-LLM-Solver
 * Tests performance under various load conditions
 */

test.describe('Load Testing Suite', () => {
  
  test.beforeEach(async ({ page }) => {
    // Set longer timeout for load tests
    test.setTimeout(300000); // 5 minutes
    
    // Navigate to the homepage
    await page.goto('/');
  });

  test('Homepage Load Performance', async ({ page }) => {
    const startTime = performance.now();
    
    // Navigate to homepage and wait for load
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const endTime = performance.now();
    const loadTime = endTime - startTime;
    
    // Performance assertions
    expect(loadTime).toBeLessThan(3000); // Less than 3 seconds
    
    // Check for essential elements
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('[data-testid="navigation"]')).toBeVisible();
  });

  test('User Registration Performance', async ({ page }) => {
    const startTime = performance.now();
    
    // Navigate to registration page
    await page.goto('/auth/register');
    
    // Fill out registration form
    await page.fill('[name="username"]', `testuser_${Date.now()}`);
    await page.fill('[name="email"]', `test_${Date.now()}@example.com`);
    await page.fill('[name="password"]', 'TestPassword123!');
    await page.fill('[name="confirm_password"]', 'TestPassword123!');
    
    // Submit form and measure response time
    await page.click('[type="submit"]');
    await page.waitForURL('**/auth/profile');
    
    const endTime = performance.now();
    const registrationTime = endTime - startTime;
    
    // Performance assertions
    expect(registrationTime).toBeLessThan(5000); // Less than 5 seconds
  });

  test('Certificate Generation Performance', async ({ page }) => {
    // First register and login a test user
    await page.goto('/auth/register');
    const username = `perftest_${Date.now()}`;
    await page.fill('[name="username"]', username);
    await page.fill('[name="email"]', `${username}@example.com`);
    await page.fill('[name="password"]', 'TestPassword123!');
    await page.fill('[name="confirm_password"]', 'TestPassword123!');
    await page.click('[type="submit"]');
    
    // Navigate to certificate generation
    await page.goto('/');
    
    const startTime = performance.now();
    
    // Fill in certificate generation form
    await page.fill('[name="system_description"]', 'dx/dt = -x + u');
    await page.fill('[name="initial_conditions"]', 'x(0) = 1');
    await page.fill('[name="domain_bounds"]', 'x ∈ [-5, 5]');
    
    // Submit certificate generation request
    await page.click('[data-testid="generate-certificate"]');
    
    // Wait for result (with timeout)
    await page.waitForSelector('[data-testid="certificate-result"]', { timeout: 120000 });
    
    const endTime = performance.now();
    const generationTime = endTime - startTime;
    
    // Performance assertions
    expect(generationTime).toBeLessThan(60000); // Less than 60 seconds
    
    // Verify certificate was generated
    const result = await page.locator('[data-testid="certificate-result"]').textContent();
    expect(result).toBeTruthy();
  });

  test('Concurrent User Simulation', async ({ browser }) => {
    const concurrentUsers = 5;
    const promises = [];
    
    for (let i = 0; i < concurrentUsers; i++) {
      promises.push(simulateUser(browser, i));
    }
    
    const startTime = performance.now();
    const results = await Promise.all(promises);
    const endTime = performance.now();
    
    const totalTime = endTime - startTime;
    const averageTime = totalTime / concurrentUsers;
    
    // All users should complete successfully
    results.forEach(result => {
      expect(result.success).toBe(true);
      expect(result.time).toBeLessThan(30000); // Each user completes in under 30s
    });
    
    // Average time should be reasonable
    expect(averageTime).toBeLessThan(20000); // Average under 20s
  });

  test('Database Performance Under Load', async ({ page }) => {
    // Create multiple queries to test database performance
    const queries = [];
    
    for (let i = 0; i < 10; i++) {
      queries.push(performDatabaseQuery(page, i));
    }
    
    const startTime = performance.now();
    const results = await Promise.all(queries);
    const endTime = performance.now();
    
    const totalTime = endTime - startTime;
    
    // All queries should complete successfully
    results.forEach(result => {
      expect(result.success).toBe(true);
    });
    
    // Total time should be reasonable for concurrent queries
    expect(totalTime).toBeLessThan(15000); // Under 15 seconds for 10 queries
  });

  test('Memory Usage Monitoring', async ({ page }) => {
    // Monitor memory usage during intensive operations
    const initialMemory = await page.evaluate(() => {
      return performance.memory ? performance.memory.usedJSHeapSize : 0;
    });
    
    // Perform memory-intensive operations
    for (let i = 0; i < 5; i++) {
      await page.goto('/');
      await page.goto('/auth/profile');
      await page.goto('/history');
    }
    
    const finalMemory = await page.evaluate(() => {
      return performance.memory ? performance.memory.usedJSHeapSize : 0;
    });
    
    const memoryIncrease = finalMemory - initialMemory;
    
    // Memory increase should be reasonable (less than 50MB)
    expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
  });

  test('API Rate Limiting', async ({ page }) => {
    // Test rate limiting by making rapid API calls
    const requests = [];
    const maxRequests = 100;
    
    for (let i = 0; i < maxRequests; i++) {
      requests.push(
        page.request.get('/api/health').catch(err => ({ error: err.message }))
      );
    }
    
    const responses = await Promise.all(requests);
    
    // Count successful vs rate-limited responses
    const successCount = responses.filter(r => r.status && r.status() === 200).length;
    const rateLimitedCount = responses.filter(r => r.status && r.status() === 429).length;
    
    // Should have some rate limiting in effect
    expect(rateLimitedCount).toBeGreaterThan(0);
    
    // But should still allow reasonable number of requests
    expect(successCount).toBeGreaterThan(10);
  });
});

/**
 * Simulate a single user's journey through the application
 */
async function simulateUser(browser, userId) {
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    const startTime = performance.now();
    
    // User journey: register -> login -> generate certificate -> view history
    await page.goto('/auth/register');
    
    const username = `loadtest_user_${userId}_${Date.now()}`;
    await page.fill('[name="username"]', username);
    await page.fill('[name="email"]', `${username}@example.com`);
    await page.fill('[name="password"]', 'TestPassword123!');
    await page.fill('[name="confirm_password"]', 'TestPassword123!');
    await page.click('[type="submit"]');
    
    // Generate a certificate
    await page.goto('/');
    await page.fill('[name="system_description"]', `dx/dt = -${userId}*x + u`);
    await page.click('[data-testid="generate-certificate"]');
    await page.waitForSelector('[data-testid="certificate-result"]', { timeout: 60000 });
    
    // Check history
    await page.goto('/history');
    
    const endTime = performance.now();
    
    await context.close();
    
    return {
      success: true,
      time: endTime - startTime,
      userId
    };
  } catch (error) {
    await context.close();
    return {
      success: false,
      error: error.message,
      userId
    };
  }
}

/**
 * Perform a database query operation
 */
async function performDatabaseQuery(page, queryId) {
  try {
    const startTime = performance.now();
    
    // Make API call that involves database query
    const response = await page.request.get(`/api/user/certificates?page=${queryId}`);
    
    const endTime = performance.now();
    
    return {
      success: response.status() === 200,
      time: endTime - startTime,
      queryId
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      queryId
    };
  }
} 