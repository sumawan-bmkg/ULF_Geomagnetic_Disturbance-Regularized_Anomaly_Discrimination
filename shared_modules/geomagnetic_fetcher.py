"""
Geomagnetic Data Fetcher
Mengambil data dari server geomagnetik via SSH dan memproses untuk incremental learning.
Menggunakan unified binary reader dari read_mdata.py
"""
import os
import sys
import paramiko
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import logging
import math

logger = logging.getLogger(__name__)

# Import unified binary reader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from read_mdata import read_604rcsv_new_python

# SSH Configuration
SSH_CONFIG = {
    'host': '202.90.198.224',
    'port': 4343,
    'username': 'precursor',
    'password': 'otomatismon'
}

class GeomagneticDataFetcher:
    """Fetch and process geomagnetic data from remote server."""
    
    def __init__(self, ssh_config=None):
        """Initialize fetcher with SSH configuration."""
        self.config = ssh_config or SSH_CONFIG
        self.ssh_client = None
        self.sftp_client = None
    def connect(self):
        """Establish SSH connection to remote server."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.ssh_client.connect(
                hostname=self.config['host'],
                port=self.config['port'],
                username=self.config['username'],
                password=self.config['password'],
                timeout=30
            )
            
            self.sftp_client = self.ssh_client.open_sftp()
            logger.info(f"Connected to {self.config['host']}:{self.config['port']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close SSH connection."""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        logger.info("Disconnected from server")
    
    def fetch_data(self, date, station='GTO'):
        """
        Fetch geomagnetic data for specific date and station.
        
        Args:
            date: datetime object or string 'YYYY-MM-DD'
            station: Station code (default: 'GTO' - recommended station)
            
        Returns:
            dict with data components or None if failed
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        year = date.year
        month = date.month
        day = date.day
        
        # Format: S{yy}{mm}{dd}.{stn} (no .gz extension)
        yy = year % 100
        mm = month
        dd = day
        
        filename = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
        # Path format as specified by user
        remote_path = f"/home/precursor/SEISMO/DATA/{station}/SData/{yy:02d}{mm:02d}/{filename}"
        
        logger.info(f"Fetching {remote_path}")
        
        try:
            # Download file to memory (no gzip decompression needed)
            with BytesIO() as file_buffer:
                self.sftp_client.getfo(remote_path, file_buffer)
                file_buffer.seek(0)
                
                # Read binary data directly
                binary_data = file_buffer.read()
                
                # Parse binary data
                data = self.parse_binary_data(binary_data, year, month, day, station)
                
                logger.info(f"Successfully fetched data for {date.date()}")
                return data
                
        except FileNotFoundError:
            logger.warning(f"File not found: {remote_path}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def parse_binary_data(self, binary_data, year, month, day, station):
        """
        Parse binary data using CORRECT format from read_binary_data.md
        
        Format:
        - Header: 32 bytes
        - Records: 17 bytes each × 86,400 (sequential, no timestamps)
        - Each record: H(2), D(2), Z(2), IX(2), IY(2), TempS(2), TempP(2), Voltage(2), Spare(1)
        
        Args:
            binary_data: Raw binary data
            year, month, day: Date components
            station: Station code
            
        Returns:
            dict with parsed components
        """
        num_seconds_in_day = 86400
        
        # Station baselines (from read_binary_data.md)
        baselines = {
            'GTO': {'H': 40000, 'Z': 30000},
            'GSI': {'H': 40000, 'Z': 30000},
            'ALR': {'H': 38000, 'Z': 32000},
            'AMB': {'H': 38000, 'Z': 32000},
            'BTN': {'H': 38000, 'Z': 32000},
            'CLP': {'H': 38000, 'Z': 32000},
            'TND': {'H': 40000, 'Z': 30000},
            'SCN': {'H': 38000, 'Z': 32000},
            'MLB': {'H': 38000, 'Z': 32000},
            'SBG': {'H': 38000, 'Z': 32000},
            'YOG': {'H': 38000, 'Z': 32000},
            'MJB': {'H': 38000, 'Z': 32000},
            'LWK': {'H': 38000, 'Z': 32000},
            'SMG': {'H': 38000, 'Z': 32000},
            'SKB': {'H': 38000, 'Z': 32000},
            'TRT': {'H': 38000, 'Z': 32000},
            'PLU': {'H': 38000, 'Z': 32000},
            'LWA': {'H': 38000, 'Z': 32000},
            'KPY': {'H': 38000, 'Z': 32000},
            'LPS': {'H': 38000, 'Z': 32000},
            'SRG': {'H': 38000, 'Z': 32000},
            'LUT': {'H': 38000, 'Z': 32000},
            'SMI': {'H': 38000, 'Z': 32000},
            'TNT': {'H': 38000, 'Z': 32000},
        }
        baseline = baselines.get(station, {'H': 40000, 'Z': 30000})
        
        # Initialize output dict
        data = {
            'Hcomp': np.full(num_seconds_in_day, np.nan),
            'Dcomp': np.full(num_seconds_in_day, np.nan),
            'Zcomp': np.full(num_seconds_in_day, np.nan),
            'Xcomp': np.full(num_seconds_in_day, np.nan),
            'Ycomp': np.full(num_seconds_in_day, np.nan),
            'IXcomp': np.full(num_seconds_in_day, np.nan),
            'IYcomp': np.full(num_seconds_in_day, np.nan),
            'TempS': np.full(num_seconds_in_day, np.nan),
            'TempP': np.full(num_seconds_in_day, np.nan),
            'Voltage': np.full(num_seconds_in_day, np.nan),
            'Time': np.arange(num_seconds_in_day),
            'date': datetime(year, month, day),
            'station': station
        }
        
        try:
            import struct
            
            # Skip 32-byte header
            header_size = 32
            record_size = 17
            data_start = binary_data[header_size:]
            
            logger.info(f"Binary data size: {len(binary_data)} bytes")
            logger.info(f"Header size: {header_size} bytes, Record size: {record_size} bytes")
            logger.info(f"Expected records: {len(data_start) // record_size}")
            logger.info(f"Baseline: H={baseline['H']} nT, Z={baseline['Z']} nT")
            
            # Parse records
            max_records = min(num_seconds_in_day, len(data_start) // record_size)
            
            for i in range(max_records):
                offset = i * record_size
                record = data_start[offset:offset+record_size]
                
                if len(record) < record_size:
                    break
                
                # Parse components (little-endian, signed int16)
                h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
                d_dev = struct.unpack('<h', record[2:4])[0] * 0.1
                z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
                ix_val = struct.unpack('<h', record[6:8])[0] * 0.01
                iy_val = struct.unpack('<h', record[8:10])[0] * 0.01
                temps_val = struct.unpack('<h', record[10:12])[0] * 0.1
                tempp_val = struct.unpack('<h', record[12:14])[0] * 0.1
                volt_val = struct.unpack('<h', record[14:16])[0] * 0.01
                
                # Add baseline to H and Z (CRITICAL!)
                data['Hcomp'][i] = baseline['H'] + h_dev
                data['Dcomp'][i] = d_dev
                data['Zcomp'][i] = baseline['Z'] + z_dev
                data['IXcomp'][i] = ix_val
                data['IYcomp'][i] = iy_val
                data['TempS'][i] = temps_val
                data['TempP'][i] = tempp_val
                data['Voltage'][i] = volt_val
            
            logger.info(f"Parsed {max_records} records")
            
            # Compute X and Y components from H and D
            # Assume D is in nT (not degrees), so we need to convert to angle first
            # For simplicity, we'll skip X/Y computation for now
            
            # Calculate statistics
            h_array = data['Hcomp'][~np.isnan(data['Hcomp'])]
            d_array = data['Dcomp'][~np.isnan(data['Dcomp'])]
            z_array = data['Zcomp'][~np.isnan(data['Zcomp'])]
            
            def safe_stat(values, stat_func):
                try:
                    if len(values) > 0:
                        result = stat_func(values)
                        if isinstance(result, (int, float)) and not math.isnan(result):
                            return float(result)
                    return None
                except (TypeError, ValueError):
                    return None
            
            data['stats'] = {
                'valid_samples': len(h_array),
                'coverage': len(h_array) / num_seconds_in_day * 100,
                'h_mean': safe_stat(h_array, np.mean),
                'h_std': safe_stat(h_array, np.std),
                'd_mean': safe_stat(d_array, np.mean),
                'd_std': safe_stat(d_array, np.std),
                'z_mean': safe_stat(z_array, np.mean),
                'z_std': safe_stat(z_array, np.std)
            }
            
            logger.info(f"Parsed {data['stats']['valid_samples']} samples ({data['stats']['coverage']:.1f}% coverage)")
            if data['stats']['h_mean'] is not None:
                logger.info(f"H: {data['stats']['h_mean']:.1f}±{data['stats']['h_std']:.1f} nT")
            if data['stats']['d_mean'] is not None:
                logger.info(f"D: {data['stats']['d_mean']:.1f}±{data['stats']['d_std']:.1f} nT")
            if data['stats']['z_mean'] is not None:
                logger.info(f"Z: {data['stats']['z_mean']:.1f}±{data['stats']['z_std']:.1f} nT")
            
        except Exception as e:
            logger.error(f"Error parsing binary data: {e}", exc_info=True)
            data['stats'] = {
                'valid_samples': 0,
                'coverage': 0.0,
                'h_mean': None,
                'h_std': None,
                'd_mean': None,
                'd_std': None,
                'z_mean': None,
                'z_std': None
            }
        
        return data
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def fetch_date_range(start_date, end_date, station='GTO'):
    """
    Fetch data for a date range.
    
    Args:
        start_date: Start date (datetime or 'YYYY-MM-DD')
        end_date: End date (datetime or 'YYYY-MM-DD')
        station: Station code (default: 'GTO')
        
    Returns:
        list of data dictionaries
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    results = []
    current_date = start_date
    
    with GeomagneticDataFetcher() as fetcher:
        while current_date <= end_date:
            data = fetcher.fetch_data(current_date, station)
            if data is not None:
                results.append(data)
            current_date += timedelta(days=1)
    
    return results


if __name__ == '__main__':
    # Test fetcher
    logging.basicConfig(level=logging.INFO)
    
    # Fetch single day from GTO station (recommended)
    with GeomagneticDataFetcher() as fetcher:
        data = fetcher.fetch_data('2026-01-13', 'GTO')
        if data:
            print(f"Fetched data for {data['date']}")
            print(f"Coverage: {data['stats']['coverage']:.1f}%")
            print(f"H component mean: {data['stats']['h_mean']:.2f} nT")
            print(f"Valid samples: {data['stats']['valid_samples']}")
        else:
            print("Failed to fetch data. Check connection and date.")
