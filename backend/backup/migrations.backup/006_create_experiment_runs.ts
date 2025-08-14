import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
  return knex.schema.createTable('experiment_runs', (table) => {
    table.uuid('id').primary().defaultTo(knex.raw('gen_random_uuid()'));
    table.string('name', 255).notNullable();
    table.text('description').nullable();
    table.uuid('system_spec_id').notNullable().references('id').inTable('system_specs').onDelete('CASCADE');
    table.jsonb('config_json').notNullable(); // Experiment configuration
    table.enum('status', ['running', 'completed', 'failed', 'cancelled']).notNullable().defaultTo('running');
    table.uuid('created_by').notNullable().references('id').inTable('users').onDelete('CASCADE');
    table.timestamps(true, true);
    table.timestamp('started_at').nullable();
    table.timestamp('completed_at').nullable();
    table.integer('total_attempts').notNullable().defaultTo(0);
    table.integer('successful_attempts').notNullable().defaultTo(0);
    table.jsonb('baseline_comparison_json').nullable();
    
    // Indexes
    table.index(['system_spec_id']);
    table.index(['status']);
    table.index(['created_by']);
    table.index(['created_at']);
    table.index(['name']);
  });
}

export async function down(knex: Knex): Promise<void> {
  return knex.schema.dropTable('experiment_runs');
}
