import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
  return knex.schema.createTable('audit_events', (table) => {
    table.uuid('id').primary().defaultTo(knex.raw('gen_random_uuid()'));
    table.uuid('user_id').notNullable().references('id').inTable('users').onDelete('CASCADE');
    table.string('action', 100).notNullable();
    table.string('entity_type', 50).notNullable();
    table.uuid('entity_id').notNullable();
    table.timestamp('at').notNullable().defaultTo(knex.fn.now());
    table.string('ip', 45).nullable(); // IPv6 compatible
    table.text('user_agent').nullable();
    
    // Indexes
    table.index(['user_id']);
    table.index(['action']);
    table.index(['entity_type', 'entity_id']);
    table.index(['at']);
  });
}

export async function down(knex: Knex): Promise<void> {
  return knex.schema.dropTable('audit_events');
}
