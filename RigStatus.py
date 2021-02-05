class RigStatusItem:
  def __init__(self, value):
    # print(f'initing rsv: {value}')
    if type(value['current']) is dict:
      self._allowed = []  # TODO: placeholder for nested dict...
      self._current = {k: RigStatusItem(v)
                       for k, v in value['current'].items()}
    elif 'allowedValues' in value.keys():
      self._allowed = value['allowedValues']
      self._current = value['current']
    elif type(value['current'] is bool):
      self._allowed = (True, False)
      self._current = value['current']

    self._category = value['category']
    self._mutable = value['mutable']
    self._callback = lambda x: x

  # def __getitem__(self, key):
  #   return self._current
  @property
  def current(self):
    return self._current

  def __set__(self, instance, value):
    self._current = value

  def mutable(self):
    self._mutable = True

  def immutable(self):
    self._mutable = False

  def callback(self, fun):
    self._callback = fun

  def __call__(self, state):
    # TODO: check that state is allowed and parseable!
    if not self._mutable:
      raise 'Couldn\'t set status'

    if type(self._current) is dict:
      current = self._current.copy()
      try:
        print(f'goal: {state}')
        for k, v in state.items():
          print(f'attempting to set {k} to {v}')
          current[k](v)
        self._current = current
      except:
        'Couldn\'t set sub-status'
    else:
      self._current = state

    self._callback(state)

  @property
  def allowed(self):
    if type(self._current) is dict:
      return {'allowedValues': self._allowed, 'category': self._category, 'current': {k: v.allowed for k, v in self._current.items()}, 'mutable': self._mutable}
    else:
      return {'allowedValues': self._allowed, 'category': self._category, 'current': self._current, 'mutable': self._mutable}

  @property
  def update(self):
    if type(self._current) is dict:
      return {'current': {k: v.update for k, v in self._current.items()}, 'mutable': self._mutable}
    else:
      return {'current': self._current, 'mutable': self._mutable}


class RigStatus(dict):
  def __init__(self, status):
    super().__init__()
    self._status = {k: RigStatusItem(v) for k, v in status.items()}

  def __getitem__(self, key):
    return self._status[key]

  def __setitem__(self, key,  value):
    self._status[key] = value

  @ property
  def allowed(self):
    return {k: v.allowed for k, v in self._status.items()}

  @ property
  def update(self):
    return {k: v.update for k, v in self._status.items()}
