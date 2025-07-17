"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-07-16

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create traffic_packets table
    op.create_table(
        'traffic_packets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('packet_id', sa.String(length=36), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('src_ip', sa.String(length=45), nullable=False),
        sa.Column('dst_ip', sa.String(length=45), nullable=False),
        sa.Column('src_port', sa.Integer(), nullable=False),
        sa.Column('dst_port', sa.Integer(), nullable=False),
        sa.Column('protocol', sa.String(length=10), nullable=False),
        sa.Column('packet_size', sa.Integer(), nullable=False),
        sa.Column('ttl', sa.Integer(), nullable=False),
        sa.Column('flags', JSON(), nullable=False),
        sa.Column('payload_entropy', sa.Float(), nullable=False),
        sa.Column('flow_id', sa.String(length=36), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_traffic_packets_packet_id'), 'traffic_packets', ['packet_id'], unique=True)
    
    # Create network_flows table
    op.create_table(
        'network_flows',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('flow_id', sa.String(length=36), nullable=False),
        sa.Column('src_ip', sa.String(length=45), nullable=False),
        sa.Column('dst_ip', sa.String(length=45), nullable=False),
        sa.Column('src_port', sa.Integer(), nullable=False),
        sa.Column('dst_port', sa.Integer(), nullable=False),
        sa.Column('protocol', sa.String(length=10), nullable=False),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('packet_count', sa.Integer(), nullable=False),
        sa.Column('byte_count', sa.Integer(), nullable=False),
        sa.Column('avg_packet_size', sa.Float(), nullable=False),
        sa.Column('flow_duration', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_network_flows_flow_id'), 'network_flows', ['flow_id'], unique=True)
    
    # Add foreign key constraint to traffic_packets
    op.create_foreign_key(
        'fk_traffic_packets_flow_id', 'traffic_packets', 'network_flows',
        ['flow_id'], ['flow_id']
    )
    
    # Create detection_results table
    op.create_table(
        'detection_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('detection_id', sa.String(length=36), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('packet_id', sa.String(length=36), nullable=False),
        sa.Column('flow_id', sa.String(length=36), nullable=True),
        sa.Column('is_malicious', sa.Boolean(), nullable=False),
        sa.Column('threat_score', sa.Integer(), nullable=False),
        sa.Column('attack_type', sa.String(length=20), nullable=False),
        sa.Column('detection_method', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('explanation', JSON(), nullable=False),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['packet_id'], ['traffic_packets.packet_id'], ),
        sa.ForeignKeyConstraint(['flow_id'], ['network_flows.flow_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_detection_results_detection_id'), 'detection_results', ['detection_id'], unique=True)
    
    # Create network_nodes table
    op.create_table(
        'network_nodes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('node_id', sa.String(length=36), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=False),
        sa.Column('packet_count', sa.Integer(), nullable=False, default=0),
        sa.Column('byte_count', sa.Integer(), nullable=False, default=0),
        sa.Column('connection_count', sa.Integer(), nullable=False, default=0),
        sa.Column('threat_score', sa.Integer(), nullable=False, default=0),
        sa.Column('is_malicious', sa.Boolean(), nullable=False, default=False),
        sa.Column('first_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_network_nodes_node_id'), 'network_nodes', ['node_id'], unique=True)
    op.create_index(op.f('ix_network_nodes_ip_address'), 'network_nodes', ['ip_address'], unique=True)
    
    # Create network_edges table
    op.create_table(
        'network_edges',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('edge_id', sa.String(length=36), nullable=False),
        sa.Column('source_ip', sa.String(length=45), nullable=False),
        sa.Column('target_ip', sa.String(length=45), nullable=False),
        sa.Column('flow_count', sa.Integer(), nullable=False, default=0),
        sa.Column('total_bytes', sa.Integer(), nullable=False, default=0),
        sa.Column('avg_packet_size', sa.Float(), nullable=False, default=0.0),
        sa.Column('connection_duration', sa.Float(), nullable=False, default=0.0),
        sa.Column('protocols', JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_network_edges_edge_id'), 'network_edges', ['edge_id'], unique=True)
    
    # Create system_metrics table
    op.create_table(
        'system_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('packets_processed', sa.Integer(), nullable=False),
        sa.Column('processing_latency_ms', sa.Float(), nullable=False),
        sa.Column('cpu_usage', sa.Float(), nullable=False),
        sa.Column('memory_usage', sa.Float(), nullable=False),
        sa.Column('active_connections', sa.Integer(), nullable=False),
        sa.Column('threat_level', sa.Integer(), nullable=False),
        sa.Column('malicious_packets', sa.Integer(), nullable=False),
        sa.Column('total_detections', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create simulation_runs table
    op.create_table(
        'simulation_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('simulation_id', sa.String(length=36), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('attack_type', sa.String(length=20), nullable=False),
        sa.Column('target_ip', sa.String(length=45), nullable=False),
        sa.Column('target_port', sa.Integer(), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=False),
        sa.Column('packet_rate', sa.Integer(), nullable=False),
        sa.Column('packet_size', sa.Integer(), nullable=True),
        sa.Column('num_threads', sa.Integer(), nullable=True),
        sa.Column('num_connections', sa.Integer(), nullable=True),
        sa.Column('connection_rate', sa.Integer(), nullable=True),
        sa.Column('use_https', sa.Boolean(), nullable=True),
        sa.Column('packets_sent', sa.Integer(), nullable=False, default=0),
        sa.Column('bytes_sent', sa.Integer(), nullable=False, default=0),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('elapsed_time', sa.Float(), nullable=True),
        sa.Column('current_packet_rate', sa.Integer(), nullable=True),
        sa.Column('errors', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_simulation_runs_simulation_id'), 'simulation_runs', ['simulation_id'], unique=True)


def downgrade() -> None:
    # Drop tables in reverse order to avoid foreign key constraints
    op.drop_table('simulation_runs')
    op.drop_table('system_metrics')
    op.drop_table('network_edges')
    op.drop_table('network_nodes')
    op.drop_table('detection_results')
    op.drop_table('traffic_packets')
    op.drop_table('network_flows')