import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
  return knex.schema.createTable('counterexamples', (table) => {
    table.uuid('id').primary().defaultTo(knex.raw('gen_random_uuid()'));
    table.uuid('candidate_id').notNullable().references('id').inTable('candidates').onDelete('CASCADE');
    table.jsonb('x_json').notNullable(); // State where certificate fails
    table.text('context').notNullable(); // Human-readable explanation
    table.jsonb('violation_metrics_json').nullable(); // Quantitative violation data
    table.timestamps(true, true);
    
    // Indexes
    table.index(['candidate_id']);
    table.index(['created_at']);
  });
}

export async function down(knex: Knex): Promise<void> {
  return knex.schema.dropTable('counterexamples');
}
