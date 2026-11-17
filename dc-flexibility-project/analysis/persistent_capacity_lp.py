import numpy as np
from scipy import sparse
import highspy as hs

from minimum_server_capacity import audit, capacity_matrix


class PersistentCapacityLP:
    def __init__(self, schedule, choices, pricing, lower, baseline):
        self.model = hs.Highs()
        for name, value in {'output_flag': False, 'threads': 1, 'solver': 'simplex',
                             'simplex_strategy': 4, 'presolve': 'off',
                             'primal_feasibility_tolerance': 1e-8,
                             'dual_feasibility_tolerance': 1e-8}.items():
            self.check(self.model.setOptionValue(name, value), name)
        self.hours = len(schedule.hours)
        self.fixed = schedule.fixed.copy()
        self.block_sizes = schedule.block_sizes.copy()
        self.amplitude = np.zeros(schedule.block_count)
        np.maximum.at(self.amplitude, schedule.blocks[schedule.columns], schedule.values[:, :2].max(axis=1))
        self.amplitude[self.amplitude == 0] = 1.
        keys = np.flatnonzero(choices & pricing.valid & pricing.movable[:, None])
        self.fluid_count = schedule.option_count - len(keys)
        self.lookup = np.full(choices.size, -1, dtype=np.int32)
        self.lookup[keys] = 1 + self.fluid_count + np.arange(len(keys))
        matrix = capacity_matrix(schedule)
        scales = np.r_[1., self.amplitude[schedule.blocks]]
        matrix.data /= np.repeat(scales, np.diff(matrix.indptr))
        matrix.data[matrix.indices >= 2 * self.hours] = 1.
        lp = hs.HighsLp()
        lp.num_col_, lp.num_row_ = matrix.shape[1], matrix.shape[0]
        lp.col_cost_ = np.r_[1., np.zeros(schedule.option_count)]
        lp.col_lower_ = np.r_[lower, np.zeros(schedule.option_count)]
        lp.col_upper_ = np.r_[baseline, np.full(schedule.option_count, hs.kHighsInf)]
        lp.row_lower_ = np.r_[np.full(2 * self.hours, -hs.kHighsInf), schedule.block_sizes * self.amplitude]
        lp.row_upper_ = np.r_[-schedule.fixed[:, :2].T.ravel(), schedule.block_sizes * self.amplitude]
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
        self.check(self.model.passModel(lp), 'passModel')
        # 原始时序提供可行基。新增候选只添加零值列，可以继续使用已有基进行单纯形迭代。
        basis = hs.HighsBasis()
        column_status = [hs.HighsBasisStatus.kLower] * lp.num_col_
        column_status[0] = hs.HighsBasisStatus.kBasic
        for first in np.unique(schedule.blocks, return_index=True)[1]:
            column_status[1 + first] = hs.HighsBasisStatus.kBasic
        row_status = [hs.HighsBasisStatus.kBasic] * (2 * self.hours)
        row_status.extend([hs.HighsBasisStatus.kLower] * schedule.block_count)
        row_status[int(schedule.original[:, :2].T.argmax())] = hs.HighsBasisStatus.kUpper
        basis.col_status, basis.row_status = column_status, row_status
        self.check(self.model.setBasis(basis), 'setBasis')

    @staticmethod
    def check(status, operation):
        if status == hs.HighsStatus.kError:
            raise RuntimeError(f'HiGHS 增量模型操作失败 {operation}')

    def solve(self, schedule, choices, pricing, seconds):
        np.testing.assert_allclose(schedule.fixed, self.fixed, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(schedule.block_sizes, self.block_sizes)
        keys = np.flatnonzero(choices & pricing.valid & pricing.movable[:, None])
        selected = np.zeros_like(self.lookup, dtype=bool)
        selected[keys] = True
        if np.any((self.lookup >= 0) & ~selected):
            raise ValueError('增量模型的候选集合必须包含已经建立的列')
        unknown = self.lookup[keys] < 0
        if unknown.any():
            local = self.fluid_count + np.flatnonzero(unknown)
            column_map = np.full(schedule.option_count, -1, dtype=np.int32)
            column_map[local] = np.arange(len(local))
            entries = np.flatnonzero(column_map[schedule.columns] >= 0)
            rows = np.r_[schedule.slots[entries], self.hours + schedule.slots[entries],
                         2 * self.hours + schedule.blocks[local]]
            columns = np.r_[column_map[schedule.columns[entries]], column_map[schedule.columns[entries]],
                            np.arange(len(local))]
            scale = self.amplitude[schedule.blocks[schedule.columns[entries]]]
            values = np.r_[schedule.values[entries, 0] / scale, schedule.values[entries, 1] / scale,
                           np.ones(len(local))]
            matrix = sparse.csc_matrix((values, (rows, columns)),
                                       shape=(2 * self.hours + schedule.block_count, len(local)))
            matrix.eliminate_zeros()
            first_new = self.model.getNumCol()
            self.check(self.model.addCols(len(local), np.zeros(len(local)), np.zeros(len(local)),
                       np.full(len(local), hs.kHighsInf), matrix.nnz, matrix.indptr,
                       matrix.indices, matrix.data), 'addCols')
            self.lookup[keys[unknown]] = first_new + np.arange(len(local))
        self.check(self.model.setOptionValue('time_limit', self.model.getRunTime() + seconds), 'time_limit')
        self.check(self.model.run(), 'run')
        status = self.model.getModelStatus()
        solution = self.model.getSolution()
        if status not in (hs.HighsModelStatus.kOptimal, hs.HighsModelStatus.kTimeLimit,
                          hs.HighsModelStatus.kIterationLimit) or not solution.value_valid:
            raise RuntimeError(f'增量 LP 未提供可行资源调度 {self.model.modelStatusToString(status)}')
        indices = np.r_[np.arange(1 + self.fluid_count), self.lookup[keys]]
        x = np.asarray(solution.col_value)[indices]
        x[1:] /= self.amplitude[schedule.blocks]
        audit(schedule, x[1:], x[0], False)
        dual = np.minimum(np.asarray(solution.row_dual)[:2 * self.hours].reshape(2, self.hours).T, 0)
        return x, dual
