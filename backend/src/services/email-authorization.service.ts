import { db } from '../utils/database';
import { logger } from '../utils/logger';

export class EmailAuthorizationService {
  private readonly collectionName = 'authorized_emails';

  /**
   * Check if an email is authorized to register
   */
  async isEmailAuthorized(email: string): Promise<boolean> {
    try {
      const normalizedEmail = email.toLowerCase().trim();
      
      const authorizedEmailsRef = db.collection(this.collectionName);
      const query = await authorizedEmailsRef.where('email', '==', normalizedEmail).limit(1).get();
      
      const isAuthorized = !query.empty;
      
      logger.info('Email authorization check', {
        email: normalizedEmail,
        isAuthorized,
      });
      
      return isAuthorized;
    } catch (error) {
      logger.error('Failed to check email authorization', {
        email,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      // In case of error, deny access for security
      return false;
    }
  }

  /**
   * Add an email to the authorized list
   */
  async addAuthorizedEmail(email: string, addedBy?: string): Promise<void> {
    try {
      const normalizedEmail = email.toLowerCase().trim();
      
      // Check if email is already authorized
      const exists = await this.isEmailAuthorized(normalizedEmail);
      if (exists) {
        throw new Error('Email is already authorized');
      }

      const authorizedEmailsRef = db.collection(this.collectionName);
      await authorizedEmailsRef.add({
        email: normalizedEmail,
        added_by: addedBy || 'system',
        added_at: new Date(),
      });

      logger.info('Email added to authorized list', {
        email: normalizedEmail,
        addedBy: addedBy || 'system',
      });
    } catch (error) {
      logger.error('Failed to add authorized email', {
        email,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Remove an email from the authorized list
   */
  async removeAuthorizedEmail(email: string): Promise<void> {
    try {
      const normalizedEmail = email.toLowerCase().trim();
      
      const authorizedEmailsRef = db.collection(this.collectionName);
      const query = await authorizedEmailsRef.where('email', '==', normalizedEmail).get();
      
      if (query.empty) {
        throw new Error('Email is not in the authorized list');
      }

      // Delete all matching documents (should be only one, but just in case)
      const batch = db.batch();
      query.docs.forEach((doc) => {
        batch.delete(doc.ref);
      });
      await batch.commit();

      logger.info('Email removed from authorized list', {
        email: normalizedEmail,
      });
    } catch (error) {
      logger.error('Failed to remove authorized email', {
        email,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Get all authorized emails (for admin purposes)
   */
  async getAuthorizedEmails(): Promise<Array<{ email: string; added_by: string; added_at: Date }>> {
    try {
      const authorizedEmailsRef = db.collection(this.collectionName);
      const snapshot = await authorizedEmailsRef.orderBy('added_at', 'desc').get();
      
      return snapshot.docs.map((doc) => {
        const data = doc.data();
        return {
          email: data.email,
          added_by: data.added_by,
          added_at: data.added_at?.toDate() || new Date(),
        };
      });
    } catch (error) {
      logger.error('Failed to get authorized emails', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  /**
   * Initialize the authorized emails collection with default entries
   */
  async initializeAuthorizedEmails(): Promise<void> {
    try {
      // Check if collection exists and has data
      const snapshot = await db.collection(this.collectionName).limit(1).get();
      if (!snapshot.empty) {
        logger.info('Authorized emails collection already initialized');
        return;
      }

      // Add default authorized emails
      const defaultEmails = [
        'patrick.cooper@colorado.edu',
        // Add more default emails here if needed
      ];

      const batch = db.batch();
      const authorizedEmailsRef = db.collection(this.collectionName);

      for (const email of defaultEmails) {
        const docRef = authorizedEmailsRef.doc();
        batch.set(docRef, {
          email: email.toLowerCase().trim(),
          added_by: 'system_init',
          added_at: new Date(),
        });
      }

      await batch.commit();

      logger.info('Authorized emails collection initialized', {
        defaultEmails,
      });
    } catch (error) {
      logger.error('Failed to initialize authorized emails', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }
}
