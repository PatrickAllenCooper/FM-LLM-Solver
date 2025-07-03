import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },   // Ramp up to 10 users over 30s
    { duration: '1m', target: 10 },    // Stay at 10 users for 1m
    { duration: '30s', target: 20 },   // Ramp up to 20 users over 30s
    { duration: '2m', target: 20 },    // Stay at 20 users for 2m
    { duration: '30s', target: 0 },    // Ramp down to 0 users over 30s
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.02'],   // Error rate must be below 2%
    errors: ['rate<0.02'],            // Custom error rate must be below 2%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

export default function () {
  // Test health endpoint
  let healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 200ms': (r) => r.timings.duration < 200,
  }) || errorRate.add(1);

  sleep(1);

  // Test main page
  let mainResponse = http.get(`${BASE_URL}/`);
  check(mainResponse, {
    'main page status is 200': (r) => r.status === 200,
    'main page response time < 1000ms': (r) => r.timings.duration < 1000,
    'main page contains title': (r) => r.body.includes('FM-LLM-Solver'),
  }) || errorRate.add(1);

  sleep(1);

  // Test API health endpoint
  let apiHealthResponse = http.get(`${BASE_URL}/api/health`);
  check(apiHealthResponse, {
    'API health status is 200': (r) => r.status === 200,
    'API health response time < 300ms': (r) => r.timings.duration < 300,
  }) || errorRate.add(1);

  sleep(1);

  // Test certificate generation (mock request)
  let certPayload = JSON.stringify({
    problem_description: "Test optimization problem for load testing",
    variables: ["x", "y"],
    constraints: ["x + y <= 10", "x >= 0", "y >= 0"],
    objective: "minimize x + y"
  });

  let certParams = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  let certResponse = http.post(`${BASE_URL}/api/generate-certificate`, certPayload, certParams);
  check(certResponse, {
    'certificate generation accepts request': (r) => r.status === 200 || r.status === 202,
    'certificate generation response time < 5000ms': (r) => r.timings.duration < 5000,
  }) || errorRate.add(1);

  sleep(2);

  // Test monitoring endpoint (if available)
  let metricsResponse = http.get(`${BASE_URL}/metrics`);
  check(metricsResponse, {
    'metrics endpoint accessible': (r) => r.status === 200 || r.status === 401, // May require auth
  });

  sleep(1);
}

// Setup function
export function setup() {
  console.log('Starting load test for FM-LLM-Solver');
  console.log(`Target URL: ${BASE_URL}`);
  
  // Verify the application is running
  let response = http.get(`${BASE_URL}/health`);
  if (response.status !== 200) {
    throw new Error(`Application not available at ${BASE_URL}. Status: ${response.status}`);
  }
  
  console.log('Application is available, starting load test...');
}

// Teardown function
export function teardown(data) {
  console.log('Load test completed');
}
