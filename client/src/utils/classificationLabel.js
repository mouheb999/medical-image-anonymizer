export const getClassificationLabel = (classification) => {
  if (!classification) return 'Unknown'
  
  const lower = classification.toLowerCase()
  
  if (lower.includes('chest'))    return 'Chest X-Ray'
  if (lower.includes('dental'))   return 'Dental X-Ray'
  if (lower.includes('pelvic'))   return 'Pelvic X-Ray'
  if (lower.includes('skull'))    return 'Skull X-Ray'
  if (lower.includes('spine') || lower.includes('vertebr'))   return 'Spine X-Ray'
  if (lower.includes('knee'))     return 'Knee X-Ray'
  if (lower.includes('hand') || lower.includes('wrist'))      return 'Hand X-Ray'
  if (lower.includes('foot') || lower.includes('ankle'))      return 'Foot X-Ray'
  if (lower.includes('shoulder')) return 'Shoulder X-Ray'
  if (lower.includes('hip'))      return 'Hip X-Ray'
  if (lower.includes('elbow'))    return 'Elbow X-Ray'
  if (lower.includes('mri'))      return 'MRI Scan'
  if (lower.includes('ct') || lower.includes('computed'))     return 'CT Scan'
  if (lower.includes('ultrasound') || lower.includes('echo')) return 'Ultrasound'
  if (lower.includes('mammograph'))  return 'Mammography'
  if (lower.includes('non') || lower.includes('not medical')) return 'Non-Medical'
  if (lower.includes('other'))    return 'Other Medical'
  if (lower.includes('accepted')) return 'Medical Image'
  
  return 'Medical Image'
}
