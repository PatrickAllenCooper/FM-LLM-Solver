import winston from 'winston';

// Create logger configuration
const loggerConfig = {
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'fm-llm-solver' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
};

// Add file transport in production
if (process.env.NODE_ENV === 'production') {
  (loggerConfig.transports as winston.transport[]).push(
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  );
}

export const logger = winston.createLogger(loggerConfig);

// Create a stream interface for integration with Express/HTTP loggers
export const loggerStream = {
  write: (message: string) => {
    logger.info(message.trim());
  }
};