#!/usr/bin/env python3
"""
Initialize security features for FM-LLM Solver web interface.
Creates database tables and sets up initial admin user.
"""

import os
import sys
import getpass

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from web_interface.app import app, db
from web_interface.models import User
from web_interface.auth import generate_api_key

def init_security():
    """Initialize security features."""
    with app.app_context():
        # Create all database tables
        print("Creating database tables...")
        db.create_all()
        print("✓ Database tables created")
        
        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if admin:
            print("! Admin user already exists")
            reset = input("Do you want to reset the admin password? (y/n): ").lower()
            if reset == 'y':
                password = getpass.getpass("Enter new admin password: ")
                confirm = getpass.getpass("Confirm password: ")
                
                if password != confirm:
                    print("✗ Passwords do not match!")
                    return
                
                admin.set_password(password)
                db.session.commit()
                print("✓ Admin password updated")
        else:
            # Create admin user
            print("\nCreating admin user...")
            
            email = input("Enter admin email: ").strip()
            if not email:
                email = "admin@example.com"
                print(f"Using default email: {email}")
            
            password = getpass.getpass("Enter admin password: ")
            confirm = getpass.getpass("Confirm password: ")
            
            if password != confirm:
                print("✗ Passwords do not match!")
                return
            
            admin = User(
                username='admin',
                email=email,
                role='admin',
                is_active=True,
                is_verified=True,
                daily_request_limit=1000  # Higher limit for admin
            )
            admin.set_password(password)
            
            # Generate API key for admin
            admin.api_key = generate_api_key()
            
            db.session.add(admin)
            db.session.commit()
            
            print("\n✓ Admin user created successfully!")
            print(f"  Username: admin")
            print(f"  Email: {email}")
            print(f"  API Key: {admin.api_key}")
            print("\n⚠️  Please save the API key securely. It won't be shown again.")
        
        # Create demo user (optional)
        create_demo = input("\nCreate demo user for testing? (y/n): ").lower()
        if create_demo == 'y':
            demo = User.query.filter_by(username='demo').first()
            if not demo:
                demo = User(
                    username='demo',
                    email='demo@example.com',
                    role='user',
                    is_active=True,
                    is_verified=True,
                    daily_request_limit=10  # Lower limit for demo
                )
                demo.set_password('demo123!')
                db.session.add(demo)
                db.session.commit()
                print("✓ Demo user created (username: demo, password: demo123!)")
            else:
                print("! Demo user already exists")
        
        print("\n✓ Security initialization complete!")
        print("\nNext steps:")
        print("1. Start the web interface: python run_web_interface.py")
        print("2. Login at: http://localhost:5000/auth/login")
        print("3. Access admin panel at: http://localhost:5000/auth/admin/users")

if __name__ == "__main__":
    print("FM-LLM Solver Security Initialization")
    print("=" * 40)
    init_security() 