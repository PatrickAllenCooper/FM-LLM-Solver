import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
  return knex.schema.createTable('system_specs', (table) => {
    table.uuid('id').primary().defaultTo(knex.raw('gen_random_uuid()'));
    table.string('name', 255).notNullable();
    table.text('description').nullable();
    table.enum('system_type', ['continuous', 'discrete', 'hybrid']).notNullable();
    table.integer('dimension').notNullable();
    table.jsonb('dynamics_json').notNullable();
    table.jsonb('constraints_json').nullable();
    table.jsonb('initial_set_json').nullable();
    table.jsonb('unsafe_set_json').nullable();
    table.uuid('created_by').notNullable().references('id').inTable('users').onDelete('CASCADE');
    table.timestamps(true, true);
    table.string('spec_version', 50).notNullable();
    table.string('spec_hash', 64).notNullable(); // SHA256 hash for reproducibility
    
    // Indexes
    table.index(['created_by']);
    table.index(['system_type']);
    table.index(['dimension']);
    table.index(['spec_hash']);
    table.index(['created_at']);
    table.index(['name']);
  });
}

export async function down(knex: Knex): Promise<void> {
  return knex.schema.dropTable('system_specs');
}
