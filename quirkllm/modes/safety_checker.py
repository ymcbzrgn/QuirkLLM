"""
Safety Checker - Critical pattern detection and risk assessment.

The SafetyChecker validates actions before execution, detecting:
- Critical patterns (rm -rf /, fork bombs, system corruption)
- High-risk patterns (chmod 777, curl | bash, etc.)
- Protected paths (/etc, /sys, /usr, /bin, etc.)
- Suspicious operations (bulk deletions, permission changes)

Used by YAMI mode for auto-execute validation and by other modes
for risk assessment and user warnings.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from quirkllm.modes.base import ActionRequest


@dataclass
class ValidationResult:
    """
    Result of safety validation.
    
    Attributes:
        is_safe: Whether action passed all safety checks
        risk_score: Risk score 0-100 (0=safe, 100=critical)
        severity: Severity level (safe, low, medium, high, critical)
        warnings: List of warning messages
        blocked_reasons: List of reasons why action was blocked
        matched_patterns: List of patterns that matched
    """
    
    is_safe: bool
    risk_score: int
    severity: str  # safe, low, medium, high, critical
    warnings: List[str] = field(default_factory=list)
    blocked_reasons: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)


class SafetyChecker:
    """
    Safety validation for actions.
    
    Checks actions against critical patterns, high-risk patterns,
    protected paths, and suspicious behaviors.
    
    Pattern Categories:
    - Critical: Immediate system destruction (rm -rf /, fork bomb)
    - High Risk: Dangerous but recoverable (chmod 777, curl | bash)
    - Medium Risk: Potentially dangerous (bulk operations)
    - Protected Paths: System directories that shouldn't be modified
    """
    
    # Critical patterns - immediate blocking
    CRITICAL_PATTERNS = [
        r'rm\s+-rf\s+/',  # rm -rf /
        r'rm\s+-rf\s+\*',  # rm -rf *
        r':\(\)\{.*:\|:.*\};',  # Fork bomb
        r'dd\s+if=/dev/zero\s+of=/dev/sd',  # Disk wipe
        r'mkfs\.',  # Format filesystem
        r'>\s*/dev/sd',  # Write to disk device
        r'chmod\s+-R\s+000',  # Remove all permissions recursively
        r'chown\s+-R\s+root:root\s+/',  # Change ownership of root
        r'killall\s+-9\s+',  # Kill all processes
        r'pkill\s+-9',  # Kill processes by name
        r'shutdown\s+',  # System shutdown
        r'reboot\s+',  # System reboot
        r'halt\s+',  # System halt
        r'init\s+0',  # Shutdown
        r'init\s+6',  # Reboot
    ]
    
    # High-risk patterns - warnings and confirmation
    HIGH_RISK_PATTERNS = [
        r'curl\s+.*\|\s*bash',  # Pipe curl to bash
        r'wget\s+.*\|\s*sh',  # Pipe wget to shell
        r'chmod\s+777',  # World writable
        r'chmod\s+-R\s+777',  # Recursive world writable
        r'sudo\s+rm',  # Root deletion
        r'sudo\s+chmod',  # Root permission change
        r'eval\s+\$\(',  # Eval command substitution
        r'exec\s+.*bash',  # Execute bash
        r'nc\s+-e',  # Netcat backdoor
        r'python.*-c.*exec',  # Python exec
        r'perl.*-e',  # Perl one-liner
        r'find\s+.*-exec\s+rm',  # Find and delete
        r'xargs\s+rm',  # Xargs delete
        r'mv\s+.*\s+/dev/null',  # Move to null
        r'>\s*/etc/',  # Write to /etc
    ]
    
    # Medium-risk patterns - warnings
    MEDIUM_RISK_PATTERNS = [
        r'rm\s+-rf\s+\w+',  # Recursive delete
        r'git\s+push\s+.*--force',  # Force push
        r'npm\s+install\s+.*-g',  # Global npm install
        r'pip\s+install\s+.*--break-system-packages',  # Break system packages
        r'docker\s+run\s+.*--privileged',  # Privileged container
        r'chmod\s+[0-7]{3}',  # Permission change
        r'chown\s+',  # Ownership change
        r'ln\s+-s',  # Symlink creation
    ]
    
    # Protected system paths
    PROTECTED_PATHS = {
        '/etc',
        '/sys',
        '/proc',
        '/dev',
        '/boot',
        '/bin',
        '/sbin',
        '/usr/bin',
        '/usr/sbin',
        '/lib',
        '/lib64',
        '/var/log',
        '/root',
    }
    
    # Sensitive user paths
    SENSITIVE_PATHS = {
        '~/.ssh',
        '~/.gnupg',
        '~/.config',
        '~/.bashrc',
        '~/.zshrc',
        '~/.profile',
    }
    
    def __init__(self):
        """Initialize SafetyChecker."""
        self.critical_regex = [re.compile(p, re.IGNORECASE) for p in self.CRITICAL_PATTERNS]
        self.high_risk_regex = [re.compile(p, re.IGNORECASE) for p in self.HIGH_RISK_PATTERNS]
        self.medium_risk_regex = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_RISK_PATTERNS]
    
    def validate_action(self, action: ActionRequest) -> ValidationResult:
        """
        Validate an action for safety.
        
        Checks action against all safety rules and returns detailed
        validation result with risk score and warnings.
        
        Args:
            action: Action to validate
            
        Returns:
            ValidationResult with safety assessment
        """
        warnings = []
        blocked_reasons = []
        matched_patterns = []
        risk_score = 0
        
        # Get action content to check
        target = action.target
        details = action.details or {}
        details_str = str(details)
        command = details.get("command", "")
        
        # Check for critical patterns
        critical_match = self._check_patterns(
            target, command, details_str, self.critical_regex, "critical"
        )
        if critical_match:
            matched_patterns.extend(critical_match)
            blocked_reasons.append(
                f"Critical pattern detected: {', '.join(critical_match)}"
            )
            risk_score = 100
            return ValidationResult(
                is_safe=False,
                risk_score=100,
                severity="critical",
                warnings=warnings,
                blocked_reasons=blocked_reasons,
                matched_patterns=matched_patterns,
            )
        
        # Check for high-risk patterns
        high_risk_match = self._check_patterns(
            target, command, details_str, self.high_risk_regex, "high-risk"
        )
        if high_risk_match:
            matched_patterns.extend(high_risk_match)
            warnings.append(
                f"High-risk pattern detected: {', '.join(high_risk_match)}"
            )
            risk_score = max(risk_score, 75)
        
        # Check for medium-risk patterns
        medium_risk_match = self._check_patterns(
            target, command, details_str, self.medium_risk_regex, "medium-risk"
        )
        if medium_risk_match:
            matched_patterns.extend(medium_risk_match)
            warnings.append(
                f"Medium-risk pattern detected: {', '.join(medium_risk_match)}"
            )
            risk_score = max(risk_score, 50)
        
        # Check protected paths
        if action.action_type in ("edit_file", "delete_file", "create_file"):
            path_result = self._check_protected_paths(target)
            if path_result:
                warnings.append(path_result)
                risk_score = max(risk_score, 60)
        
        # Determine severity
        if risk_score >= 75:
            severity = "high"
        elif risk_score >= 50:
            severity = "medium"
        elif risk_score >= 25:
            severity = "low"
        else:
            severity = "safe"
        
        # Action is safe if no critical blocks and risk score < 100
        is_safe = len(blocked_reasons) == 0
        
        return ValidationResult(
            is_safe=is_safe,
            risk_score=risk_score,
            severity=severity,
            warnings=warnings,
            blocked_reasons=blocked_reasons,
            matched_patterns=matched_patterns,
        )
    
    def _check_patterns(
        self,
        target: str,
        command: str,
        details: str,
        patterns: List[re.Pattern],
        category: str,
    ) -> List[str]:
        """
        Check text against pattern list.
        
        Args:
            target: Action target
            command: Command string
            details: Details string
            patterns: Compiled regex patterns
            category: Pattern category name
            
        Returns:
            List of matched pattern descriptions
        """
        matches = []
        combined_text = f"{target} {command} {details}"
        
        for pattern in patterns:
            if pattern.search(combined_text):
                matches.append(f"{category}: {pattern.pattern}")
        
        return matches
    
    def _check_protected_paths(self, path: str) -> Optional[str]:
        """
        Check if path is in protected directories.
        
        Args:
            path: Path to check
            
        Returns:
            Warning message if protected, None otherwise
        """
        try:
            # Try to resolve path, but handle non-existent paths
            try:
                path_str = str(Path(path).resolve())
            except (OSError, RuntimeError):
                # If resolve fails, use absolute path
                path_str = str(Path(path).absolute())
        except Exception:
            # Fallback to string comparison
            path_str = path
        
        # Check protected system paths
        for protected in self.PROTECTED_PATHS:
            # Check both original and with /private prefix (macOS symlink)
            if path_str.startswith(protected) or path_str.startswith(f"/private{protected}"):
                return f"Protected system path: {protected}"
        
        # Check sensitive user paths
        for sensitive in self.SENSITIVE_PATHS:
            try:
                sensitive_expanded = str(Path(sensitive).expanduser())
                if path_str.startswith(sensitive_expanded):
                    return f"Sensitive user path: {sensitive}"
            except Exception:
                continue
        
        return None
    
    def is_critical(self, action: ActionRequest) -> bool:
        """
        Quick check if action contains critical patterns.
        
        Args:
            action: Action to check
            
        Returns:
            True if critical pattern detected
        """
        result = self.validate_action(action)
        return result.severity == "critical"
    
    def is_high_risk(self, action: ActionRequest) -> bool:
        """
        Quick check if action is high risk.
        
        Args:
            action: Action to check
            
        Returns:
            True if high risk or critical
        """
        result = self.validate_action(action)
        return result.severity in ("high", "critical")
    
    def get_risk_score(self, action: ActionRequest) -> int:
        """
        Get risk score for action.
        
        Args:
            action: Action to score
            
        Returns:
            Risk score 0-100
        """
        result = self.validate_action(action)
        return result.risk_score
