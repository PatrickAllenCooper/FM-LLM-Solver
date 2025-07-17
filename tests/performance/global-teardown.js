// global-teardown.js
const { chromium } = require('@playwright/test');
const fs = require('fs').promises;
const path = require('path');

async function globalTeardown() {
  console.log('🧹 Starting performance test environment cleanup...');
  
  try {
    // Generate performance report
    await generatePerformanceReport();
    
    // Clean up test data
    await cleanupTestData();
    
    // Archive test results
    await archiveTestResults();
    
    console.log('✅ Performance test cleanup complete');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error.message);
    // Don't throw error to avoid breaking CI
  }
}

async function generatePerformanceReport() {
  console.log('📊 Generating performance report...');
  
  try {
    const resultsDir = 'test-results';
    const reportPath = path.join(resultsDir, 'performance-report.json');
    
    // Check if results directory exists
    try {
      await fs.access(resultsDir);
    } catch {
      console.log('ℹ️  No test results directory found');
      return;
    }
    
    // Read test results
    let testResults = {};
    try {
      const resultsFile = path.join(resultsDir, 'results.json');
      const resultsData = await fs.readFile(resultsFile, 'utf8');
      testResults = JSON.parse(resultsData);
    } catch (error) {
      console.log('ℹ️  No JSON results file found');
    }
    
    // Generate performance summary
    const performanceReport = {
      timestamp: new Date().toISOString(),
      environment: {
        baseURL: process.env.BASE_URL || 'http://localhost:5000',
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
      },
      summary: {
        totalTests: testResults.suites?.length || 0,
        totalDuration: testResults.stats?.duration || 0,
        passedTests: testResults.stats?.passed || 0,
        failedTests: testResults.stats?.failed || 0,
        skippedTests: testResults.stats?.skipped || 0,
      },
      recommendations: generateRecommendations(testResults),
      thresholds: {
        pageLoadTime: 3000,
        apiResponseTime: 1000,
        certificateGenerationTime: 60000,
        concurrentUserLimit: 10,
      }
    };
    
    // Write performance report
    await fs.writeFile(reportPath, JSON.stringify(performanceReport, null, 2));
    console.log(`✅ Performance report generated: ${reportPath}`);
    
  } catch (error) {
    console.log('⚠️  Could not generate performance report:', error.message);
  }
}

async function cleanupTestData() {
  console.log('🗑️  Cleaning up test data...');
  
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    const baseURL = process.env.BASE_URL || 'http://localhost:5000';
    
    // Login as admin to clean up test users
    await page.goto(`${baseURL}/auth/login`);
    await page.fill('[name="email"]', 'perftest@example.com');
    await page.fill('[name="password"]', 'TestPassword123!');
    await page.click('[type="submit"]');
    
    // Check if we have admin access
    await page.goto(`${baseURL}/admin`);
    
    // If admin panel is accessible, clean up test users
    const adminPanelExists = await page.locator('[data-testid="admin-panel"]').isVisible().catch(() => false);
    
    if (adminPanelExists) {
      console.log('🔧 Cleaning up test users via admin panel...');
      
      // Delete test users created during load testing
      const testUserPatterns = ['loadtest_user_', 'perftest_', 'testuser_'];
      
      for (const pattern of testUserPatterns) {
        try {
          // This would depend on your actual admin interface
          await page.fill('[data-testid="user-search"]', pattern);
          await page.click('[data-testid="search-users"]');
          
          // Select and delete test users (implementation depends on your UI)
          const deleteButtons = await page.locator('[data-testid="delete-user"]').all();
          for (const button of deleteButtons) {
            await button.click();
            // Confirm deletion if needed
            await page.click('[data-testid="confirm-delete"]').catch(() => {});
          }
        } catch (error) {
          console.log(`⚠️  Could not clean up users with pattern ${pattern}`);
        }
      }
    } else {
      console.log('ℹ️  No admin access available for cleanup');
    }
    
  } catch (error) {
    console.log('⚠️  Could not clean up test data:', error.message);
  } finally {
    await browser.close();
  }
}

async function archiveTestResults() {
  console.log('📦 Archiving test results...');
  
  try {
    const resultsDir = 'test-results';
    const archiveDir = 'test-archives';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const archivePath = path.join(archiveDir, `performance-${timestamp}`);
    
    // Create archive directory
    await fs.mkdir(archiveDir, { recursive: true });
    await fs.mkdir(archivePath, { recursive: true });
    
    // Copy test results to archive
    try {
      const files = await fs.readdir(resultsDir);
      for (const file of files) {
        const sourcePath = path.join(resultsDir, file);
        const destPath = path.join(archivePath, file);
        await fs.copyFile(sourcePath, destPath);
      }
      
      console.log(`✅ Test results archived to: ${archivePath}`);
    } catch (error) {
      console.log('ℹ️  No test results to archive');
    }
    
    // Clean up old archives (keep last 10)
    try {
      const archives = await fs.readdir(archiveDir);
      const sortedArchives = archives
        .filter(name => name.startsWith('performance-'))
        .sort()
        .reverse();
      
      if (sortedArchives.length > 10) {
        const archivesToDelete = sortedArchives.slice(10);
        for (const archive of archivesToDelete) {
          await fs.rmdir(path.join(archiveDir, archive), { recursive: true });
        }
        console.log(`🗑️  Cleaned up ${archivesToDelete.length} old archives`);
      }
    } catch (error) {
      console.log('⚠️  Could not clean up old archives');
    }
    
  } catch (error) {
    console.log('⚠️  Could not archive test results:', error.message);
  }
}

function generateRecommendations(testResults) {
  const recommendations = [];
  
  // Analyze test results and generate recommendations
  if (testResults.stats) {
    const { failed, total, duration } = testResults.stats;
    
    if (failed > 0) {
      recommendations.push({
        type: 'performance',
        severity: 'high',
        message: `${failed} performance tests failed. Review failing tests and optimize accordingly.`
      });
    }
    
    if (duration > 300000) { // 5 minutes
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: 'Performance test suite takes longer than 5 minutes. Consider optimizing test execution.'
      });
    }
    
    if (total < 10) {
      recommendations.push({
        type: 'coverage',
        severity: 'low',
        message: 'Consider adding more performance test scenarios for better coverage.'
      });
    }
  }
  
  return recommendations;
}

module.exports = globalTeardown; 