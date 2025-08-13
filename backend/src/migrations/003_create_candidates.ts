import type { Knex } from 'knex';

export async function up(knex: Knex): Promise<void> {
  return knex.schema.createTable('candidates', (table) => {
    table.uuid('id').primary().defaultTo(knex.raw('gen_random_uuid()'));
    table.uuid('system_spec_id').notNullable().references('id').inTable('system_specs').onDelete('CASCADE');
    table.enum('certificate_type', ['lyapunov', 'barrier', 'inductive_invariant']).notNullable();
    table.enum('generation_method', ['llm', 'sos', 'sdp', 'quadratic_template']).notNullable();
    table.string('llm_provider', 50).nullable();
    table.string('llm_model', 100).nullable();
    table.enum('llm_mode', ['direct_expression', 'basis_coeffs', 'structure_constraints']).nullable();
    table.jsonb('llm_config_json').nullable();
    table.text('candidate_expression').notNullable();
    table.jsonb('candidate_json').notNullable();
    table.enum('verification_status', ['pending', 'verified', 'failed', 'timeout']).notNullable().defaultTo('pending');
    table.decimal('margin', 15, 8).nullable();
    table.uuid('created_by').notNullable().references('id').inTable('users').onDelete('CASCADE');
    table.timestamps(true, true);
    table.timestamp('verified_at').nullable();
    table.integer('generation_duration_ms').nullable();
    table.integer('verification_duration_ms').nullable();
    
    // Indexes
    table.index(['system_spec_id']);
    table.index(['certificate_type']);
    table.index(['generation_method']);
    table.index(['verification_status']);
    table.index(['created_by']);
    table.index(['created_at']);
    table.index(['llm_provider', 'llm_model']);
  });
}

export async function down(knex: Knex): Promise<void> {
  return knex.schema.dropTable('candidates');
}
