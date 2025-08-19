const { initializeApp, getApps } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const bcrypt = require('bcryptjs');

// Initialize Firebase Admin SDK
let app;
if (!getApps().length) {
  app = initializeApp({
    projectId: process.env.GOOGLE_CLOUD_PROJECT || 'fmgen-net-production',
  });
} else {
  app = getApps()[0];
}

const db = getFirestore(app);

async function createAdminUser() {
  try {
    const email = 'patrick.cooper@colorado.edu';
    const password = 'admin123'; // You can change this
    const role = 'admin';

    // Check if user already exists
    const usersRef = db.collection('users');
    const existingUserQuery = await usersRef.where('email', '==', email).limit(1).get();

    if (!existingUserQuery.empty) {
      // Update existing user to admin
      const userDoc = existingUserQuery.docs[0];
      await userDoc.ref.update({
        role: 'admin',
        updated_at: new Date(),
      });
      console.log(`✅ Updated existing user ${email} to admin role`);
    } else {
      // Create new admin user
      const password_hash = await bcrypt.hash(password, 12);
      
      const newUserData = {
        email,
        password_hash,
        role,
        created_at: new Date(),
        updated_at: new Date(),
      };

      const docRef = await usersRef.add(newUserData);
      console.log(`✅ Created new admin user ${email} with ID: ${docRef.id}`);
      console.log(`🔑 Password: ${password}`);
    }

    console.log('✨ Admin setup complete!');
  } catch (error) {
    console.error('❌ Error creating admin user:', error);
  }
}

createAdminUser();

