import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
  return knex.schema.createTable('users', (table) => {
    table.uuid('id').primary().defaultTo(knex.raw('gen_random_uuid()'));
    table.string('email', 255).notNullable().unique();
    table.string('password_hash', 255).notNullable();
    table.enum('role', ['admin', 'researcher', 'viewer']).notNullable().defaultTo('researcher');
    table.timestamps(true, true);
    table.timestamp('last_login_at').nullable();
    
    // Indexes
    table.index(['email']);
    table.index(['role']);
    table.index(['created_at']);
  });
}

export async function down(knex: Knex): Promise<void> {
  return knex.schema.dropTable('users');
}
