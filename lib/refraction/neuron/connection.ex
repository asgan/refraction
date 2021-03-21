defmodule Refraction.Neuron.Connection do
  alias Refraction.Neuron.Connection

  @type t :: %__MODULE__{
          pid: pid(),
          source_pid: pid(),
          target_pid: pid(),
          weight: float
        }
  defstruct pid: nil,
            source_pid: nil,
            target_pid: nil,
            weight: 0.382

  @spec start_link(map()) :: {:ok, pid()}
  def start_link(attributes \\ %{}) do
    {:ok, pid} = Agent.start_link(fn -> %Connection{} end)
    update(pid, Map.merge(attributes, %{pid: pid}))
    {:ok, pid}
  end

  @spec get(pid()) :: Connection.t()
  def get(pid), do: Agent.get(pid, fn connection -> connection end)

  @spec update(pid(), map()) :: :ok
  def update(pid, new_attributes) do
    Agent.update(pid, fn current_attributes ->
      Map.merge(current_attributes, new_attributes)
    end)
  end

  @spec connection_for(pid(), pid()) :: {:ok, pid()}
  def connection_for(source_pid, target_pid) do
    {:ok, pid} = start_link()
    update(pid, %{source_pid: source_pid, target_pid: target_pid})
    {:ok, pid}
  end
end
