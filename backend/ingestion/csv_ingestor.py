"""
CSV file traffic ingestor
"""
import os
import csv
from typing import Iterator, Dict, Any, List
from datetime import datetime
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_ingestor import BaseTrafficIngestor
from models.data_models import TrafficPacket, ProtocolType
from core.exceptions import TrafficIngestionError


class CsvIngestor(BaseTrafficIngestor):
    """Ingest traffic from CSV files with feature normalization"""
    
    def __init__(self):
        super().__init__("CSV File")
        self.required_columns = [
            'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
            'protocol', 'packet_size', 'ttl', 'payload_entropy'
        ]
        self.optional_columns = ['flags', 'packet_id', 'is_malicious']
    
    def validate_source(self, source: str) -> bool:
        """Validate CSV file exists and has required columns"""
        if not os.path.exists(source):
            raise TrafficIngestionError(f"CSV file not found: {source}")
        
        if not os.path.isfile(source):
            raise TrafficIngestionError(f"Source is not a file: {source}")
        
        if not source.lower().endswith('.csv'):
            self.logger.warning(f"File extension may not be CSV: {source}")
        
        try:
            # Read first few rows to validate structure
            df = pd.read_csv(source, nrows=5)
            
            if df.empty:
                raise TrafficIngestionError(f"CSV file is empty: {source}")
            
            # Check for required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise TrafficIngestionError(
                    f"CSV file missing required columns: {missing_columns}. "
                    f"Available columns: {list(df.columns)}"
                )
            
            self.logger.info(f"CSV file validated. Columns: {list(df.columns)}")
            
        except pd.errors.EmptyDataError:
            raise TrafficIngestionError(f"CSV file is empty or invalid: {source}")
        except Exception as e:
            raise TrafficIngestionError(f"Cannot read CSV file {source}: {e}")
        
        return True
    
    def ingest(self, source: str, **kwargs) -> Iterator[TrafficPacket]:
        """
        Ingest packets from CSV file
        
        Args:
            source: Path to CSV file
            **kwargs: Additional parameters
                - max_packets: Maximum number of packets to read
                - filters: Packet filtering criteria
                - normalize_features: Apply feature normalization
                - chunk_size: Number of rows to read at once
        """
        self.validate_source(source)
        self.start_ingestion()
        
        max_packets = kwargs.get('max_packets', None)
        filters = kwargs.get('filters', {})
        normalize_features = kwargs.get('normalize_features', True)
        chunk_size = kwargs.get('chunk_size', 1000)
        
        try:
            # Read CSV in chunks to handle large files
            chunk_iter = pd.read_csv(source, chunksize=chunk_size)
            
            for chunk_df in chunk_iter:
                if normalize_features:
                    chunk_df = self._normalize_features(chunk_df)
                
                for _, row in chunk_df.iterrows():
                    if max_packets and self.packets_processed >= max_packets:
                        return
                    
                    try:
                        traffic_packet = self._convert_csv_row(row)
                        
                        if traffic_packet and self.apply_filters(traffic_packet, filters):
                            traffic_packet = self.preprocess_packet(traffic_packet)
                            self.packets_processed += 1
                            yield traffic_packet
                            
                    except Exception as e:
                        self.errors_encountered += 1
                        self.logger.warning(f"Skipping invalid row: {e}")
                        continue
        
        except Exception as e:
            raise TrafficIngestionError(f"Error reading CSV file: {e}")
        
        finally:
            self.end_ingestion()
    
    def _convert_csv_row(self, row: pd.Series) -> TrafficPacket:
        """Convert CSV row to TrafficPacket"""
        try:
            # Parse timestamp
            timestamp_str = str(row['timestamp'])
            try:
                # Try ISO format first
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try common timestamp formats
                    timestamp = pd.to_datetime(timestamp_str).to_pydatetime()
                except Exception:
                    # Use current time as fallback
                    timestamp = datetime.now()
                    self.logger.warning(f"Could not parse timestamp: {timestamp_str}")
            
            # Parse protocol
            protocol_str = str(row['protocol']).upper()
            try:
                protocol = ProtocolType(protocol_str)
            except ValueError:
                # Default to TCP if protocol not recognized
                protocol = ProtocolType.TCP
                self.logger.warning(f"Unknown protocol: {protocol_str}, defaulting to TCP")
            
            # Parse flags (if present)
            flags = []
            if 'flags' in row and pd.notna(row['flags']):
                flags_str = str(row['flags'])
                if '|' in flags_str:
                    flags = flags_str.split('|')
                elif ',' in flags_str:
                    flags = flags_str.split(',')
                else:
                    flags = [flags_str] if flags_str else []
                
                # Clean up flag names
                flags = [flag.strip().upper() for flag in flags if flag.strip()]
            
            # Get packet ID (if present)
            packet_id = None
            if 'packet_id' in row and pd.notna(row['packet_id']):
                packet_id = str(row['packet_id'])
            
            return TrafficPacket(
                timestamp=timestamp,
                src_ip=str(row['src_ip']),
                dst_ip=str(row['dst_ip']),
                src_port=int(row['src_port']),
                dst_port=int(row['dst_port']),
                protocol=protocol,
                packet_size=int(row['packet_size']),
                ttl=int(row['ttl']),
                flags=flags,
                payload_entropy=float(row['payload_entropy']),
                packet_id=packet_id
            )
            
        except Exception as e:
            raise TrafficIngestionError(f"Error converting CSV row: {e}")
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature normalization to the dataframe"""
        try:
            df_normalized = df.copy()
            
            # Normalize packet size (0-1 range based on common packet sizes)
            if 'packet_size' in df_normalized.columns:
                df_normalized['packet_size'] = df_normalized['packet_size'].clip(0, 65535)
            
            # Normalize TTL (0-1 range)
            if 'ttl' in df_normalized.columns:
                df_normalized['ttl'] = df_normalized['ttl'].clip(1, 255)
            
            # Ensure entropy is in 0-1 range
            if 'payload_entropy' in df_normalized.columns:
                df_normalized['payload_entropy'] = df_normalized['payload_entropy'].clip(0.0, 1.0)
            
            # Normalize port numbers
            for port_col in ['src_port', 'dst_port']:
                if port_col in df_normalized.columns:
                    df_normalized[port_col] = df_normalized[port_col].clip(0, 65535)
            
            return df_normalized
            
        except Exception as e:
            self.logger.warning(f"Feature normalization failed: {e}")
            return df
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Get mapping of expected column names to alternative names"""
        return {
            'timestamp': ['time', 'ts', 'datetime'],
            'src_ip': ['source_ip', 'src_addr', 'source_address'],
            'dst_ip': ['dest_ip', 'dst_addr', 'destination_ip', 'destination_address'],
            'src_port': ['source_port', 'sport'],
            'dst_port': ['dest_port', 'dport', 'destination_port'],
            'protocol': ['proto', 'ip_proto'],
            'packet_size': ['size', 'length', 'packet_length'],
            'ttl': ['time_to_live'],
            'payload_entropy': ['entropy', 'data_entropy'],
            'flags': ['tcp_flags', 'flag'],
            'packet_id': ['id', 'pkt_id']
        }
    
    def auto_map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically map column names to expected format"""
        column_mapping = self.get_column_mapping()
        df_mapped = df.copy()
        
        for expected_col, alternatives in column_mapping.items():
            if expected_col not in df_mapped.columns:
                for alt_col in alternatives:
                    if alt_col in df_mapped.columns:
                        df_mapped = df_mapped.rename(columns={alt_col: expected_col})
                        self.logger.info(f"Mapped column '{alt_col}' to '{expected_col}'")
                        break
        
        return df_mapped